import torch
import wandb

from models import utils
from tools.trainer_base import TrainerBase


class TrainerClassification(TrainerBase):
    def __init__(self, conf, now=None, k_fold=0):
        super(TrainerClassification, self).__init__(conf, now=now, k_fold=k_fold)

    def _train(self, epoch):
        self.model.train()

        batch_loss_y = 0
        batch_loss_t = 0
        for iteration, data in enumerate(self.loader_train.Loader):
            x_in = data["input"]
            target = data["label"]

            x_in = x_in.to(self.device)
            target = target.to(self.device)

            if (x_in.shape[0] / torch.cuda.device_count()) <= torch.cuda.device_count():
                break  # avoid BN issue

            output = self.model(x_in)
            result_key = [key for key in output if "result_" in key][0]

            # compute loss
            loss_y = self.criterion(output[result_key], target)
            if "loss_t" in output.keys():
                loss_t = output["loss_t"] * 1e-2
                batch_loss_t += loss_t.item()
                loss = loss_y + loss_t
            else:
                loss = loss_y

            batch_loss_y += loss_y.item()

            if not torch.isfinite(loss):
                raise Exception("Loss is NAN. End training.")

            # ----- backward ----- #
            self.optimizer.zero_grad()
            self.accelerator.backward(loss)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            if (iteration + 1) % self._validate_interval == 0:
                self._validate(epoch)

            # compute metric
            output_argmax = (
                torch.argmax(output[result_key], dim=1).detach().cpu().numpy()
            )
            target_argmax = torch.argmax(target.squeeze(), dim=1).detach().cpu().numpy()
            self.metric_train.update(output_argmax, target_argmax)

        metric_result = self.metric_train.get_results()

        metric_dict = {}
        metric_dict["lr"] = self.get_learning_rate()
        metric_dict["acc"] = metric_result["acc"]
        metric_dict["f1"] = metric_result["f1"]
        metric_dict["loss_y"] = batch_loss_y / self.loader_train.Loader.__len__()
        metric_dict["loss_t"] = batch_loss_t / self.loader_train.Loader.__len__()

        utils.log_epoch("train", epoch, metric_dict, self.conf["env"]["wandb"])
        self.metric_train.reset()

    def _validate(self, epoch):
        self.model.eval()

        for iteration, data in enumerate(self.loader_valid.Loader):
            with torch.no_grad():
                x_in = data["input"]
                target = data["label"]

                x_in = x_in.to(self.device)
                target = target.to(self.device)

                output = self.model(x_in)
                result_key = [key for key in output if "result_" in key][0]

                # compute metric
                output_argmax = (
                    torch.argmax(output[result_key], dim=1).detach().cpu().numpy()
                )
                target_argmax = (
                    torch.argmax(target.squeeze(), dim=1).detach().cpu().numpy()
                )
                self.metric_val.update(output_argmax, target_argmax)

        metric_result = self.metric_val.get_results()

        metric_dict = {}
        metric_dict["acc"] = metric_result["acc"]
        metric_dict["f1"] = metric_result["f1"]

        utils.log_epoch("validation", epoch, metric_dict, self.conf["env"]["wandb"])
        self.check_metric(epoch, metric_dict)
        self.metric_val.reset()

    def run(self):
        for epoch in range(1, self.conf["env"]["epoch"] + 1):
            self._train(epoch)
            self._validate(epoch)

            if (epoch - self.last_saved_epoch) > self.conf["env"]["early_stop_epoch"]:
                utils.Logger().info(
                    "The model seems to be converged. Early stop training."
                )
                utils.Logger().info(f'Best f1 -----> {self.metric_best["f1"]}')
                if self.conf["env"]["wandb"]:
                    wandb.log({"Best f1": self.metric_best["f1"]})
                break
