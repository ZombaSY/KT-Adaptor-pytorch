import copy
from collections import OrderedDict

import torch
import torch.nn as nn

from models import utils
from models.backbones import MobileOne, timm_backbone
from models.heads import MLP
from models.necks import KTAdaptor


class ConvNextv2_base_backbone(nn.Module):
    def __init__(self, conf_model):
        super().__init__()
        self.backbone = timm_backbone.BackboneLoader(
            "convnextv2_base.fcmae_ft_in22k_in1k_384", exportable=True, pretrained=True
        )

        if conf_model["saved_ckpt"] != "":
            self.load_pretrained(conf_model["saved_ckpt"])

    def freeze_backbone(self):
        self.backbone.requires_grad = False
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        utils.Logger().info("freezed backbone")

    def load_pretrained(self, pretrained_path):
        self.load_state_dict(
            torch.load(pretrained_path, weights_only=True), strict=False
        )
        utils.Logger().info(f"loaded {self.__class__.__name__}.")

    def forward_backbone(self, x):
        return self.backbone(x)


class Mobileone_s0_base_backbone(nn.Module):
    def __init__(self, conf_model):
        super().__init__()
        self.backbone = MobileOne.mobileone(variant="s0")

        if conf_model["saved_ckpt"] != "":
            self.load_pretrained(conf_model["saved_ckpt"])

    def freeze_backbone(self):
        self.backbone.requires_grad = False
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        utils.Logger().info("freezed backbone")

    def load_pretrained(self, pretrained_path):
        self.load_state_dict(
            torch.load(pretrained_path, weights_only=True), strict=False
        )
        utils.Logger().info(f"loaded {self.__class__.__name__}.")

    def load_pretrained_imagenet(self, dst):
        pretrained_states = torch.load(dst)
        pretrained_states_backbone = OrderedDict()

        for item in pretrained_states.keys():
            if item in ["linear.weight", "linear.bias"]:
                continue
            pretrained_states_backbone[item] = pretrained_states[item]

        self.backbone.load_state_dict(pretrained_states_backbone)

    def forward_backbone(self, x):
        return self.backbone(x)


class Swin_t_base_backbone(nn.Module):
    def __init__(self, conf_model):
        super().__init__()
        self.backbone = timm_backbone.BackboneLoader(
            "swin_large_patch4_window7_224.ms_in22k_ft_in1k",
            exportable=True,
            pretrained=True,
        )

        if conf_model["saved_ckpt"] != "":
            self.load_pretrained(conf_model["saved_ckpt"])

    def freeze_backbone(self):
        self.backbone.requires_grad = False
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        utils.Logger().info("freezed backbone")

    def load_pretrained(self, pretrained_path):
        self.load_state_dict(
            torch.load(pretrained_path, weights_only=True), strict=False
        )
        utils.Logger().info(f"loaded {self.__class__.__name__}.")

    def forward_backbone(self, x):
        latent = self.backbone(x)
        return latent.permute(0, 3, 1, 2)


class KTAdaptor_base(nn.Module):
    def __init__(self, conf_model):
        super().__init__()
        self.conf_model = conf_model
        self.target_dim = conf_model["target_dim"]

        # init and freeze backbones
        if conf_model["inference_mode"] is False:
            self.models_list = conf_model["facial_tasks"].keys()
            self.tasks = nn.ModuleList(
                [
                    globals()[model](conf_model["facial_tasks"][model])
                    for model in self.models_list
                ]
            )
            control_vec = []
            for idx, task in enumerate(self.tasks):
                task.backbone.requires_grad = False
                task.backbone.eval()
                for param in task.backbone.parameters():
                    param.requires_grad = False

                if list(self.models_list)[idx] == conf_model["selected_task"]:
                    control_vec.append(True)
                    if conf_model["freeze_canonical_backbone"] is False:
                        for param in task.backbone.parameters():
                            param.requires_grad = True
                else:
                    control_vec.append(False)

                utils.Logger().info("freezed backbone")

            self.register_buffer("control_vec", torch.tensor(control_vec))
            self.task_idx = torch.where(self.control_vec is True)[0]
            self.backbone = self.tasks[self.task_idx].backbone
            self.head = self.tasks[self.task_idx].head
        else:
            task = globals()[conf_model["selected_task"]](conf_model)
            self.backbone = task.backbone
            self.head = task.head
            self.register_buffer(
                "control_vec", torch.zeros(conf_model["num_task"], dtype=torch.bool)
            )

        self.register_buffer("token_placeholder", torch.zeros(conf_model["token_dim"]))

    def filter_state_dict(self):
        filtered_state_dict = OrderedDict()
        selected_keys = "tasks." + str(self.task_idx)
        for k, v in self.state_dict().items():
            if "tasks." in k:
                if selected_keys in k:
                    new_k = k.replace(selected_keys, "backbone")
                    filtered_state_dict[new_k] = v
            else:
                filtered_state_dict[k] = v

        return filtered_state_dict

    def forward(self, x):
        out_dict = {}
        if self.training:
            tokens = []
            latent_task = None
            for idx, task in enumerate(self.tasks):
                latent = task.forward_backbone(x)
                if idx == self.task_idx:
                    latent_task = latent  # store for feed-forward
                token = torch._adaptive_avg_pool2d(latent, 1)
                token = torch.flatten(token, start_dim=1)

                # fill the rest of elements with zero
                token = torch.cat(
                    [
                        token,
                        self.token_placeholder[token.shape[-1] :].expand(
                            token.shape[0], -1
                        ),
                    ],
                    dim=-1,
                )
                tokens.append(token)  # Append each token to the list
            tokens_fused = torch.stack(
                tokens, dim=1
            )  # Shape: [batch_size, num_models, features]

            token_att, loss_t = self.kt_adaptor(tokens_fused, self.control_vec)
            latent_task = latent_task + (token_att.unsqueeze(-1).unsqueeze(-1))
            out = self.head(latent_task)

            out_dict["result_task"] = out
            out_dict["loss_t"] = loss_t
        else:
            if "Swin" in self.conf_model["selected_task"]:
                latent_task = self.backbone(x)
                latent_task = latent_task.permute(0, 3, 1, 2)
            else:
                latent_task = self.backbone(x)
            token = torch._adaptive_avg_pool2d(latent_task, 1)
            token = torch.flatten(token, start_dim=1)
            token = torch.cat(
                [
                    token,
                    self.token_placeholder[token.shape[-1] :].expand(
                        token.shape[0], -1
                    ),
                ],
                dim=-1,
            )
            token = token.unsqueeze(1)

            token_att = self.kt_adaptor.forward_inference(token, self.control_vec)
            latent_task_att = latent_task + (token_att.unsqueeze(-1).unsqueeze(-1))

            # swap on face recognition
            out = self.head(latent_task_att)
            # out = self.head.forward_embedding(latent_task_att)

            out_dict["result_task"] = out
            out_dict["latent_before"] = latent_task
            out_dict["latent_after"] = latent_task_att

        return out_dict


class ModelSoupsAverage(nn.Module):
    def __init__(self, conf_model):
        super().__init__()
        self.conf_model = conf_model

        # init and freeze backbones
        self.models_list = conf_model["facial_tasks"].keys()
        self.tasks = nn.ModuleList(
            [
                globals()[model](conf_model["facial_tasks"][model])
                for model in self.models_list
            ]
        )
        self.fused_backbone = None
        control_vec = []
        for idx, task in enumerate(self.tasks):
            task.backbone.requires_grad = False
            task.backbone.eval()

            if list(self.models_list)[idx] == conf_model["selected_task"]:
                control_vec.append(True)
            else:
                control_vec.append(False)

            # fuse backbones
            if self.fused_backbone is None:
                self.fused_backbone = copy.deepcopy(task.backbone)
            else:
                self.fused_backbone = self.average_models(
                    self.fused_backbone, task.backbone, 0.5
                )

            utils.Logger().info("averaged the weights of backbones")

        # freeze fused backbone
        for param in self.fused_backbone.parameters():
            param.requires_grad = False

        self.register_buffer("control_vec", torch.tensor(control_vec))
        self.task_idx = torch.where(self.control_vec is True)[0]
        self.head = self.tasks[self.task_idx].head

    def filter_state_dict(self):
        filtered_state_dict = OrderedDict()
        selected_keys = "tasks." + str(self.task_idx)
        for k, v in self.state_dict().items():
            if "tasks." in k:
                if selected_keys in k:
                    new_k = k.replace(selected_keys, "backbone")
                    filtered_state_dict[new_k] = v
            else:
                filtered_state_dict[k] = v

        return filtered_state_dict

    def average_models(self, model_a, model_b, alpha=0.5):
        """
        Returns a new model whose state_dict is the weighted average:
        new = alpha * A + (1 - alpha) * B
        Works for parameters and buffers (e.g., BatchNorm running stats).
        """
        model_new = copy.deepcopy(model_a)

        sd_a = model_a.state_dict()
        sd_b = model_b.state_dict()
        sd_new = {}

        # Sanity check: same keys
        if sd_a.keys() != sd_b.keys():
            missing = sd_a.keys() ^ sd_b.keys()
            raise ValueError(f"State dicts don't match. Diff keys: {sorted(missing)}")

        with torch.no_grad():
            for k in sd_a.keys():
                ta, tb = sd_a[k], sd_b[k]
                if torch.is_floating_point(ta) and torch.is_floating_point(tb):
                    tb = tb.to(device=ta.device, dtype=ta.dtype)
                    sd_new[k] = alpha * ta + (1.0 - alpha) * tb
                else:
                    # non-float tensors (e.g., num_batches_tracked) – pick from A
                    sd_new[k] = ta.clone()

            model_new.load_state_dict(sd_new, strict=True)

        return model_new

    def forward(self, x):
        out_dict = {}

        # forward
        latent_task = self.fused_backbone(x)
        if "Swin" in self.conf_model["selected_task"]:
            latent_task = latent_task.permute(0, 3, 1, 2)
        out = self.head(latent_task)

        out_dict["result_task"] = out
        out_dict["latent_task"] = latent_task

        return out_dict


class ModelEnsemblePool(nn.Module):
    def __init__(self, conf_model):
        super().__init__()
        self.conf_model = conf_model

        # init and freeze backbones
        self.models_list = conf_model["facial_tasks"].keys()
        self.tasks = nn.ModuleList(
            [
                globals()[model](conf_model["facial_tasks"][model])
                for model in self.models_list
            ]
        )
        control_vec = []
        for idx, task in enumerate(self.tasks):
            task.backbone.requires_grad = False
            task.backbone.eval()

            for param in task.backbone.parameters():
                param.requires_grad = False

            if list(self.models_list)[idx] == conf_model["selected_task"]:
                control_vec.append(True)
            else:
                control_vec.append(False)

        self.register_buffer("control_vec", torch.tensor(control_vec))
        self.task_idx = torch.where(self.control_vec is True)[0]
        self.head = self.tasks[self.task_idx].head

    def forward(self, x):
        out_dict = {}

        latent_task = None
        for task in self.tasks:
            if latent_task is None:
                latent_task = task.forward_backbone(x)
            else:
                # average the feature latents
                latent_task = (latent_task + task.forward_backbone(x)) / 2

        out = self.head(latent_task)

        out_dict["result_task"] = out
        out_dict["latent_task"] = latent_task

        return out_dict


class ModelEnsembleCat(nn.Module):
    def __init__(self, conf_model):
        super().__init__()
        self.conf_model = conf_model

        # init and freeze backbones
        self.models_list = conf_model["facial_tasks"].keys()
        self.tasks = nn.ModuleList(
            [
                globals()[model](conf_model["facial_tasks"][model])
                for model in self.models_list
            ]
        )
        control_vec = []
        for idx, task in enumerate(self.tasks):
            task.backbone.requires_grad = False
            task.backbone.eval()

            for param in task.backbone.parameters():
                param.requires_grad = False

            if list(self.models_list)[idx] == conf_model["selected_task"]:
                control_vec.append(True)
            else:
                control_vec.append(False)

        self.register_buffer("control_vec", torch.tensor(control_vec))
        self.task_idx = torch.where(self.control_vec is True)[0]

        self.head = getattr(MLP, type(self.tasks[self.task_idx].head).__name__)(
            conf_model["channel_in"], conf_model["num_class"]
        )

    def forward(self, x):
        out_dict = {}

        latent_task = None
        for task in self.tasks:
            if latent_task is None:
                latent_task = task.forward_backbone(x)
            else:
                latent_task = torch.cat([latent_task, task.forward_backbone(x)], dim=1)

        out = self.head(latent_task)

        out_dict["result_task"] = out
        out_dict["latent_task"] = latent_task

        return out_dict


class ConvNextv2_face_age_estimation(ConvNextv2_base_backbone):
    def __init__(self, conf_model):
        super().__init__(conf_model)
        self.head = MLP.SimpleRegressor(
            channel_in=1024, num_class=conf_model["num_class"]
        )

    def forward(self, x):
        out_dicts = {}

        latent = self.forward_backbone(x)
        results = self.head(latent)

        out_dicts["latent_age"] = latent
        out_dicts["result_age"] = results

        return out_dicts


class ConvNextv2_face_emotion_recognition(ConvNextv2_base_backbone):
    def __init__(self, conf_model):
        super().__init__(conf_model)
        self.head = MLP.SimpleClassifier(
            in_features=1024, num_class=conf_model["num_class"]
        )

    def forward(self, x):
        out_dicts = {}

        latent = self.forward_backbone(x)
        results = self.head(latent)

        out_dicts["latent_fer"] = latent
        out_dicts["result_fer"] = results

        return out_dicts


class ConvNextv2_face_landmark_detection(ConvNextv2_base_backbone):
    def __init__(self, conf_model):
        super().__init__(conf_model)
        self.head = MLP.SimpleLandmarker(
            channel_in=1024, num_points=conf_model["num_class"], bottleneck_size=[2, 2]
        )

    def forward(self, x):
        out_dicts = {}

        latent = self.forward_backbone(x)
        results = self.head(latent)

        out_dicts["latent_lm"] = latent
        out_dicts["result_lm"] = results

        return out_dicts


class ConvNextv2_face_recognition(ConvNextv2_base_backbone):
    def __init__(self, conf_model):
        super().__init__(conf_model)
        self.head = MLP.SimpleRecognizer(
            channel_in=1024, num_class=conf_model["num_class"]
        )
        self.inference_mode = conf_model["inference_mode"]

    def forward(self, x):
        out_dicts = {}

        latent = self.forward_backbone(x)
        if not self.inference_mode:
            results = self.head(latent)
        else:
            results = self.head.forward_embedding(latent)

        out_dicts["latent_fr"] = latent
        out_dicts["result_fr"] = results

        return out_dicts


class Mobileone_s0_age_estimation(Mobileone_s0_base_backbone):
    def __init__(self, conf_model):
        super().__init__(conf_model)
        self.head = MLP.SimpleLandmarker(
            channel_in=1024, num_points=conf_model["num_class"], bottleneck_size=[2, 2]
        )

    def forward(self, x):
        out_dict = {}

        latent = self.forward_backbone(x)
        results = self.head(latent)
        out_dict["latent_age"] = latent
        out_dict["result_age"] = results

        return out_dict


class Mobileone_s0_face_emotion_recognition(Mobileone_s0_base_backbone):
    def __init__(self, conf_model):
        super().__init__(conf_model)
        self.head = MLP.SimpleLandmarker(
            channel_in=1024, num_points=conf_model["num_class"], bottleneck_size=[2, 2]
        )

    def load_pretrained_imagenet(self, dst):
        pretrained_states = torch.load(dst)
        pretrained_states_backbone = OrderedDict()

        for item in pretrained_states.keys():
            if item in ["linear.weight", "linear.bias"]:
                continue
            pretrained_states_backbone[item] = pretrained_states[item]

        self.backbone.load_state_dict(pretrained_states_backbone)

    def forward(self, x):
        out_dict = {}

        latent = self.forward_backbone(x)
        results = self.head(latent)
        out_dict["latent_fer"] = latent
        out_dict["result_fer"] = results

        return out_dict


class Mobileone_s0_landmark_detection(Mobileone_s0_base_backbone):
    def __init__(self, conf_model):
        super().__init__(conf_model)
        self.head = MLP.SimpleLandmarker(
            channel_in=1024, num_points=conf_model["num_class"], bottleneck_size=[2, 2]
        )

    def forward(self, x):
        out_dict = {}

        latent = self.forward_backbone(x)
        results = self.head(latent)
        out_dict["latent_lm"] = latent
        out_dict["result_lm"] = results

        return out_dict


class Mobileone_s0_face_recognition(Mobileone_s0_base_backbone):
    def __init__(self, conf_model):
        super().__init__(conf_model)
        self.head = MLP.SimpleRecognizer(
            channel_in=1024, num_class=conf_model["num_class"]
        )
        self.inference_mode = conf_model["inference_mode"]

    def forward(self, x):
        out_dicts = {}

        latent = self.forward_backbone(x)
        if not self.inference_mode:
            results = self.head(latent)
        else:
            results = self.head.forward_embedding(latent)

        out_dicts["latent_fr"] = latent
        out_dicts["result_fr"] = results

        return out_dicts


class Swin_t_face_age_estimation(Swin_t_base_backbone):
    def __init__(self, conf_model):
        super().__init__(conf_model)
        self.head = MLP.SimpleRegressor(
            channel_in=1536, num_class=conf_model["num_class"]
        )

    def forward(self, x):
        out_dicts = {}

        latent = self.forward_backbone(x)
        results = self.head(latent)

        out_dicts["latent_age"] = latent
        out_dicts["result_age"] = results

        return out_dicts


class Swin_t_face_emotion_recognition(Swin_t_base_backbone):
    def __init__(self, conf_model):
        super().__init__(conf_model)
        self.head = MLP.SimpleClassifier(
            in_features=1536, num_class=conf_model["num_class"]
        )

    def forward(self, x):
        out_dicts = {}

        latent = self.forward_backbone(x)
        results = self.head(latent)

        out_dicts["latent_fer"] = latent
        out_dicts["result_fer"] = results

        return out_dicts


class Swin_t_face_landmark_detection(Swin_t_base_backbone):
    def __init__(self, conf_model):
        super().__init__(conf_model)
        self.head = MLP.SimpleLandmarker(
            channel_in=1536, num_points=conf_model["num_class"], bottleneck_size=[2, 2]
        )

    def forward(self, x):
        out_dicts = {}

        latent = self.forward_backbone(x)
        results = self.head(latent)

        out_dicts["latent_lm"] = latent
        out_dicts["result_lm"] = results

        return out_dicts


class Swin_t_face_recognition(Swin_t_base_backbone):
    def __init__(self, conf_model):
        super().__init__(conf_model)
        self.head = MLP.SimpleRecognizer(
            channel_in=1536, num_class=conf_model["num_class"]
        )
        self.inference_mode = conf_model["inference_mode"]

    def forward(self, x):
        out_dicts = {}

        latent = self.forward_backbone(x)
        if not self.inference_mode:
            results = self.head(latent)
        else:
            results = self.head.forward_embedding(latent)

        out_dicts["latent_fr"] = latent
        out_dicts["result_fr"] = results

        return out_dicts


class KTAdaptorModel(KTAdaptor_base):
    def __init__(self, conf_model):
        super().__init__(conf_model)
        self.kt_adaptor = KTAdaptor.KTAdaptor(
            n_tasks=conf_model["num_task"],
            in_dims=conf_model["token_dim"],
            depths=conf_model["depths"],
            num_heads=conf_model["num_heads"],
            target_dim=conf_model["target_dim"],
        )
