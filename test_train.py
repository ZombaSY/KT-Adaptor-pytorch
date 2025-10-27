import os
import torch
import random
import numpy as np
import cv2
import importlib
import logging

from accelerate import Accelerator
from models import lr_scheduler
from tools.trainer_base import TrainerBase
from models import utils
from thop import profile

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"


class ModelMini:
    def __init__(self):
        self.device = 'cuda'
        self.accelerator = Accelerator()

        self.x = torch.rand([8, 3, 224, 224]).to(self.device)
        self.y = torch.rand([8, 196]).to(self.device).long()
        self.x_add = torch.tensor([True, True, True, True]).to(self.device)
        self.x_add.requires_grad = False

        # select model configuraiton and load from configs for test
        spec = importlib.util.spec_from_file_location("conf", 'configs/train_knowledge_token.py')
        imported_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(imported_module)
        self.conf = imported_module.conf
        utils.Logger('log.txt', level=logging.DEBUG if self.conf['env']['debug'] else logging.INFO)

        self.model = TrainerBase.init_model(self.conf , self.device)
        self.model.to(self.device)
        self.criterion = TrainerBase.init_criterion(self.conf['criterion'], self.device)
        self.optimizer = TrainerBase.init_optimizer(self.conf['optimizer'], self.model)

        self.epochs = 2
        self.steps = 5
        self.t_max = 20
        self.cycles = self.epochs / self.t_max
        self.scheduler = lr_scheduler.WarmupCosineSchedule(optimizer=self.optimizer,
                                                           warmup_steps=self.steps * 20,
                                                           t_total=self.epochs * self.steps,
                                                           cycles=self.cycles,
                                                           last_epoch=-1)

        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

        # torch.save(self.model.module.filter_state_dict(), 'model.pt')
        # self.train_mini()
        # self.inference()
        self.measure_computations()
        # self.measure_inference_time()
        # self.measure_memory_consumption()
        # self.save_jit()

    def load_image(self, src):
        img = cv2.imread(src, cv2.IMREAD_COLOR) / 255
        img = cv2.resize(img, (512, 512))
        img = (img - 0.5) / 0.25

        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

    def train_mini(self):
        self.model.train()
        for epoch in range(self.epochs):
            for step in range(self.steps):
                out = self.model(self.x)
                key = list(filter(lambda x: True if 'result_' in x else False, out.keys()))[0]
                loss = self.criterion(out[key], self.y)
                self.accelerator.backward(loss)
                self.optimizer.zero_grad()
                self.optimizer.step()
                self.scheduler.step()
                print(loss.item())

    def inference(self):
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.x)

    def measure_inference_time(self):
        # measure inference time
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        timings = np.zeros((self.steps - 1, 1))
        self.model.eval()
        with torch.no_grad():
            for step in range(self.steps):
                starter.record()
                _ = self.model(self.x)
                if step != 0:
                    ender.record()
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender)
                    timings[step - 1] = curr_time
        mean_syn = np.sum(timings) / (self.steps - 1)
        print(f'{mean_syn}ms')

    def measure_computations(self):
        self.model.eval()
        # Calculate FLOPs and params
        macs, params = profile(self.model.module, inputs=([self.x]))

        print(f"Total FLOPs: {macs*2/1e9:.2f}G")  # MACs * 2 = FLOPs
        print(f"Total Parameters: {params/1e6:.2f}M")

    def measure_memory_consumption(self):
        self.model.eval()
        with torch.no_grad():
            print('Max Memory Allocated: ', torch.cuda.max_memory_allocated())

    def save_jit(self):
        with torch.no_grad():
            self.model.eval()
            m = torch.jit.script(self.model)
            torch.jit.save(m, 'model.torchscript')


def main():
    seed = 3407
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    ModelMini()


if __name__ == '__main__':
    main()
