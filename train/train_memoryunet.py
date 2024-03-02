import os
import torch
from torch import optim
from ours.diffusion.models import GaussianDiffusion, MemoryUnet
from ours.diffusion.respace import SpacedDiffusion, iter_denoise
from data.TJ_PV600 import trainloader, testloader
from tqdm import tqdm
from time import time
from utils import *


class Trainer:
    def __init__(
            self,
            unet_pred="eps",
            lr=2e-4,
    ):
        super(Trainer, self).__init__()
        self.device = "cuda"
        self.lr = lr
        self.unet_pred = unet_pred
        self.diffsuion = GaussianDiffusion(mean_type=unet_pred)
        self.unet = MemoryUnet(
            in_channels=1,
            model_channels=64,
            out_channels=1,
            channel_mult=(1, 2, 2, 4, 4),
            num_res_blocks=2,
            num_heads=8,
            num_memories=600
        ).to(self.device)
        self.save_path = os.path.join(get_project_path(project_name="Defense"), "pretrained", "Memory_Unet_for_DM.pth")
        self.unet.load_state_dict(torch.load(self.save_path))

        self.trainloader = trainloader
        self.testloader = testloader

    def train(self, epochs):
        start = time()
        optimizer = optim.AdamW(self.unet.parameters(), lr=self.lr, weight_decay=0.005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)
        for epoch in range(epochs):
            count = len(self.trainloader)
            epoch_loss = 0
            self.unet.train()
            for step, (img, _) in tqdm(enumerate(self.trainloader), desc=f"train step {epoch+1}/{epochs}", total=count):
                optimizer.zero_grad()
                img = img.to(self.device)
                batch_size = img.shape[0]

                t = self.diffsuion.get_rand_t(batch_size, self.device, min_t=0, max_t=1000)

                loss = self.diffsuion.training_losses(model=self.unet, x_start=img, t=t)
                epoch_loss += loss
                loss.backward()
                optimizer.step()

            if optimizer.state_dict()['param_groups'][0]['lr'] > self.lr / 1e2:
                scheduler.step()

            torch.save(self.unet.state_dict(), self.save_path)

            print(f"Epoch:{epoch+1}/{epochs}  Loss:{epoch_loss:.8f}")

            if epoch % 10 == 0 and epoch != 0:
                self.test_ddim()

        end = time()
        seconds = int(end - start)
        minutes = seconds // 60
        remain_second = seconds % 60
        print(f"time consumed: {minutes}min{remain_second}s")

    def test_ddim(self):
        self.unet.eval()
        spacediffusion = SpacedDiffusion(num_ddpm_timesteps=1000, num_ddim_timesteps=100)
        imgs, _ = next(iter(self.testloader))
        imgs = imgs.to(self.device)

        # torch.manual_seed(10)
        noise = torch.randn_like(imgs)

        final_sample = spacediffusion.ddim_sample_loop(self.unet, shape=imgs.shape, noise=noise, progress=True)

        draw_img_groups([imgs, final_sample])

    def test_ddpm(self):
        self.unet.eval()
        imgs, _ = next(iter(self.testloader))
        imgs = imgs.to(self.device)
        noise = torch.randn_like(imgs)
        final_sample = self.diffsuion.p_sample_loop(self.unet, shape=imgs.shape, noise=noise, progress=True)
        draw_img_groups([imgs, final_sample])

    def denoise(self, t=50):
        assert t % 10 == 0
        self.unet.eval()
        imgs, _ = next(iter(self.testloader))
        imgs = imgs.to(self.device)
        final_sample = iter_denoise(self.unet, imgs=imgs, t=t, sampler="ddim", progress=True)
        draw_img_groups([imgs, final_sample])


def main():
    # trainer = TrainDiffusion(unet_pred=ModelMeanType.START_X)
    trainer = Trainer()
    # trainer.train(1000)
    # trainer.test_ddim()
    # trainer.test_ddpm()
    trainer.denoise(t=50)
if __name__ == "__main__":
    main()



