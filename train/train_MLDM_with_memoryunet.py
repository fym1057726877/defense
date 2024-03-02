import os
import torch
from tqdm import tqdm
from time import time
from torch import optim
from ours.diffusion.models import GaussianDiffusion, MemoryUnet
from ours.diffusion.respace import SpacedDiffusion, iter_denoise
from data.TJ_PV600 import trainloader, testloader
from utils import get_project_path, draw_img_groups, print_time_consumed
from LDM.aemodel.autoencoder import MemoryAutoencoder


class TrainDiffusion:
    def __init__(
            self,
            lr=2e-4,
    ):
        super(TrainDiffusion, self).__init__()
        self.device = "cuda"
        self.lr = lr
        self.diffsuion = GaussianDiffusion(mean_type="eps")
        self.unet = MemoryUnet(
            in_channels=4,
            model_channels=32,
            out_channels=4,
            channel_mult=(1, 2, 4),
            num_res_blocks=2,
            num_heads=8,
            num_memories=600,
            features=2048,
        ).to(self.device)
        self.save_path = os.path.join(
            get_project_path(project_name="Defense"), "pretrained", "Memory_Unet_for_LDM.pth")
        self.unet.load_state_dict(torch.load(self.save_path))

        self.autoencoder = MemoryAutoencoder(
            in_ch=1,
            ch=128,
            z_ch=4,
            out_ch=1,
            embed_dim=4,
            ch_mult=(1, 2, 4),
            resolution=64,
            num_res_blocks=2,
            attn_resolutions=(4, 2, 1),
            dropout=0.0,
            double_z=False,
            num_memories=600,
            threshold=1e-4,
            recloss_type="l1"
        ).to(self.device)
        self.ae_path = os.path.join(get_project_path(project_name="Defense"), "pretrained", "MemoryAE_LDM.pth")
        self.autoencoder.load_state_dict(torch.load(self.ae_path))

        self.trainloader = trainloader
        self.testloader = testloader

        self.spacediffusion = SpacedDiffusion(num_ddpm_timesteps=1000, num_ddim_timesteps=50)

    def train(self, epochs):
        start = time()
        optimizer = optim.AdamW(self.unet.parameters(), lr=self.lr, weight_decay=0.005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)
        self.autoencoder.eval()
        for epoch in range(epochs):
            count = len(self.trainloader)
            epoch_loss = 0
            self.unet.train()
            for step, (img, _) in tqdm(enumerate(self.trainloader), desc=f"train step {epoch+1}/{epochs}", total=count):
                optimizer.zero_grad()
                img = img.to(self.device)
                batch_size = img.shape[0]

                t = self.diffsuion.get_rand_t(batch_size, device=self.device, min_t=0, max_t=1000)
                z = self.autoencoder.encode(img)

                loss = self.diffsuion.training_losses(model=self.unet, x_start=z, t=t)
                epoch_loss += loss
                loss.backward()
                optimizer.step()

            if optimizer.state_dict()['param_groups'][0]['lr'] > self.lr / 1e2:
                scheduler.step()

            torch.save(self.unet.state_dict(), self.save_path)

            print(f"Epoch:{epoch+1}/{epochs}  Loss:{epoch_loss / count:.8f}")

            if (epoch+1) % 10 == 0 and epoch != 0:
                self.test_ddim()

        end = time()
        seconds = int(end - start)
        print_time_consumed(seconds)

    def test_ddim(self):
        self.unet.eval()

        imgs, _ = next(iter(self.testloader))
        imgs = imgs.to(self.device)
        h = self.autoencoder.encode(imgs)
        z = self.autoencoder.encode_encode(h)[0]
        rec = self.autoencoder.decode(z)

        # torch.manual_seed(10)
        noise = torch.randn_like(h)
        generated_h = self.spacediffusion.ddim_sample_loop(self.unet, shape=noise.shape, noise=noise, progress=True)
        generated_z = self.autoencoder.encode_encode(generated_h)[0]
        generated_sample = self.autoencoder.decode(generated_z)
        draw_img_groups([imgs, rec, generated_sample])

    # def test_ddpm(self):
    #     self.unet.eval()
    #     imgs, _ = next(iter(self.testloader))
    #     imgs = imgs.to(self.device)
    #     z = self.encoder(imgs)
    #
    #     noise = torch.randn_like(z)
    #     final_sample = self.diffsuion.p_sample_loop(self.unet, shape=z.shape, noise=noise, progress=True)
    #     generated_sample = self.decoder(final_sample)
    #     draw_img_groups([noise, generated_sample])

    def test_iter_denoise(self, t=50):
        assert t % 10 == 0
        self.unet.eval()
        imgs, _ = next(iter(self.testloader))
        imgs = imgs.to(self.device)
        h = self.autoencoder.encode(imgs)
        z = self.autoencoder.encode_encode(h)[0]
        rec = self.autoencoder.decode(z)

        final_h = iter_denoise(self.unet, imgs=h, t=t)
        final_z = self.autoencoder.encode_encode(final_h)[0]
        final_sample = self.autoencoder.decode(final_z)
        draw_img_groups([imgs, rec, final_sample])


def main():
    trainer = TrainDiffusion()
    # trainer.train(500)
    trainer.test_ddim()
    # trainer.test_ddpm()
    # trainer.test_iter_denoise(t=50)


if __name__ == "__main__":
    main()



