import itertools
import logging
import multiprocessing
import os

import numpy as np
import pandas as pd
import torch

from models import utils
from tools.trainer_base import TrainerBase


class Inferencer:
    def __init__(self, conf):
        self.conf = conf
        self.saved_model_directory = os.path.split(self.conf["model"]["saved_ckpt"])[0]
        utils.Logger(
            "log.txt",
            level=logging.DEBUG if self.conf["env"]["debug"] else logging.INFO,
        )

        use_cuda = self.conf["env"]["cuda"] and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.loader_valid = TrainerBase.init_data_loader(
            conf=self.conf, conf_dataloader=self.conf["dataloader_valid"]
        )

        self.criterion = TrainerBase.init_criterion(self.conf["criterion"], self.device)

        self.model = TrainerBase.init_model(self.conf, self.device)
        self.model.module.load_state_dict(
            torch.load(self.conf["model"]["saved_ckpt"]), strict=False
        )
        self.model.eval()

        dir_path, fn = os.path.split(self.conf["model"]["saved_ckpt"])
        fn, ext = os.path.splitext(fn)

        self.save_dir = os.path.join(dir_path, fn)
        self.num_batches_val = int(len(self.loader_valid))
        self.metric_val = TrainerBase.init_metric(
            self.conf["env"]["task"], self.conf["model"]["num_class"]
        )

        self.image_mean = torch.tensor(self.loader_valid.image_loader.image_mean).to(
            self.device
        )
        self.image_std = torch.tensor(self.loader_valid.image_loader.image_std).to(
            self.device
        )

        self.data_stat = {}

    def inference_classification(self, epoch):
        self.model.eval()

        for iteration, data in enumerate(self.loader_valid.Loader):
            with torch.no_grad():
                x_in = data["input"]
                target = data["label"]

                x_in = x_in.to(self.device)
                target = target.long().to(
                    self.device
                )  # (shape: (batch_size, img_h, img_w))

                output = self.model(x_in)
                result_key = [key for key in output if "result_" in key][0]

                output_argmax = (
                    torch.argmax(output[result_key], dim=1).detach().cpu().numpy()
                )
                target_argmax = (
                    torch.argmax(target.squeeze(), dim=1).detach().cpu().numpy()
                    if self.conf["env"]["mode"] == "valid"
                    else np.zeros_like(output_argmax)
                )
                self.metric_val.update(output_argmax, target_argmax)

        if self.conf["env"]["mode"] == "valid":
            metric_dict = {}
            metric_result = self.metric_val.get_results()
            metric_dict["acc"] = metric_result["acc"]
            metric_dict["f1"] = metric_result["f1"]

            utils.log_epoch("validation", epoch, metric_dict, False)

            df = pd.DataFrame(
                {
                    "fn": self.loader_valid.image_loader.df["input"],
                    "label": self.metric_val.get_pred_flatten(),
                }
            )
            df.to_csv(self.save_dir + "_out.csv", encoding="utf-8-sig", index=False)
            self.metric_val.reset()

    def inference_regression(self, epoch):
        self.model.eval()
        metric_dict = {"loss": []}
        batch_losses = []

        # AE parameter
        tolerance_level = 5
        correct_predictions = 0
        total_predictions = 0

        for iteration, data in enumerate(self.loader_valid.Loader):
            with torch.no_grad():
                x_in = data["input"]
                img_id = data["input_path"]
                target = (
                    data["label"].to(self.device) if "label" in data.keys() else None
                )

                x_in = x_in.to(self.device)

                output = self.model(x_in)
                result_key = [key for key in output if "result_" in key][0]

                # compute loss
                if self.conf["env"]["mode"] == "valid":
                    loss = self.criterion(output[result_key], target)
                    batch_losses.append(loss.item())

                    if not torch.isfinite(loss):
                        raise Exception("Loss is NAN. End training.")

                    # AE metric: cumulative score
                    absolute_errors = torch.abs(
                        output[result_key].view(-1) - target.view(-1)
                    )
                    correct_predictions += (
                        (absolute_errors <= tolerance_level).sum().item()
                    )
                    total_predictions += absolute_errors.size(
                        0
                    )  # Update total predictions count

                self.__post_process(x_in, target, output[result_key], img_id, iteration)

        if self.conf["env"]["mode"] == "valid":
            batch_losses = np.array(batch_losses)
            loss_mean = batch_losses.sum() / self.loader_valid.Loader.__len__()

            metric_dict = {}
            metric_dict["loss"] = loss_mean
            metric_dict["CS"] = correct_predictions / total_predictions

            utils.log_epoch("validation", epoch, metric_dict, False)
            self.metric_val.reset()

    def inference_recognition(self, epoch):
        self.model.eval()
        metric_dict = {"loss": []}
        batch_losses = []

        # FR parameter
        embeddings_before = []
        classes = []
        with torch.no_grad():
            for iteration, data in enumerate(self.loader_valid.Loader):
                x_in = data["input"]
                target = (
                    data["label"].to(self.device) if "label" in data.keys() else None
                )

                x_in = x_in.to(self.device)

                output = self.model(x_in)

                key_result = [key for key in output if "result_" in key][0]
                embeddings_before.extend(
                    torch.nn.functional.adaptive_avg_pool2d(
                        output["latent_before"], 1
                    ).squeeze()
                )
                classes.extend(target)

                # compute loss
                if self.conf["env"]["mode"] == "valid":
                    loss = self.criterion(output[key_result], target.long())
                    batch_losses.append(loss.item())

                    if not torch.isfinite(loss):
                        raise Exception("Loss is NAN. End training.")

        if self.conf["env"]["mode"] == "valid":
            batch_losses = np.array(batch_losses)
            loss_mean = batch_losses.sum() / self.loader_valid.Loader.__len__()

            metric_dict = {}
            metric_dict["loss"] = loss_mean

            utils.log_epoch("validation", epoch, metric_dict, False)
            self.metric_val.reset()

        # Visualize embeddings using t-SNE
        if len(embeddings_before) > 0:
            all_embeddings_before = (
                torch.stack(embeddings_before, dim=-1).detach().cpu().numpy().T
            )
            np.save("visualize/data/cfp-embeddings-after.npy", all_embeddings_before)

    def __post_process_regression(self, x_img, target, output, img_id):
        for idx in range(x_img.shape[0]):
            utils.append_data_stats(self.data_stat, "img_id", img_id[idx])
            for i in range(output.shape[-1]):
                key_name = "predict_" + str(i).zfill(2)
                utils.append_data_stats(
                    self.data_stat, key_name, output[idx][i].detach().cpu().item()
                )

    def __post_process_landmark(self, x_img, target, output, img_id):
        if self.conf["env"]["draw_results"]:

            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)

            with multiprocessing.Pool(multiprocessing.cpu_count() // 2) as pools:
                x_img = (
                    utils.denormalize_img(x_img, self.image_mean, self.image_std)
                    .detach()
                    .cpu()
                    .numpy()
                )
                output_np = output.detach().cpu().numpy()

                pools.map(
                    utils.multiprocessing_wrapper,
                    zip(
                        itertools.repeat(utils.draw_landmark),
                        x_img,
                        output_np,
                        itertools.repeat(self.save_dir),
                        img_id,
                    ),
                )

    def __post_process(self, x_img, target, output, img_id, iteration):
        if self.conf["env"]["task"] == "landmark":
            self.__post_process_landmark(x_img, target, output, img_id)
        elif self.conf["env"]["task"] == "regression":
            self.__post_process_regression(x_img, target, output, img_id)

        print(
            f'iteration {iteration} -> {(iteration + 1) * self.conf["dataloader_valid"]["batch_size"]} images done !!'
        )  # TODO: last iteration is invalid

    def inference(self):
        if self.conf["env"]["task"] == "classification":
            self.inference_classification(0)
        elif self.conf["env"]["task"] in ["regression", "landmark"]:
            self.inference_regression(0)
        elif self.conf["env"]["task"] == "recognition":
            self.inference_recognition(0)

        # save meta data
        # df = pd.DataFrame(self.data_stat)
        # df.to_csv(self.save_dir + '_out.csv', encoding='utf-8-sig', index=False)
