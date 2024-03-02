import os
import torch
from tqdm import tqdm
from time import time
from torch import optim
from ContrastExperiances.Diffusion.models import GaussianDiffusion, UNetModel
from ContrastExperiances.Diffusion.respace import SpacedDiffusion
from LDM.dataset.mnist import trainloader, testloader
from utils import get_project_path, draw_img_groups
from LDM.aemodel.autoencoder import MaskMemoryAutoencoder


class TrainDiffusion:
    def __init__(
            self,
            lr=2e-4,
    ):
        super(TrainDiffusion, self).__init__()
        self.device = "cuda"
        self.lr = lr
        self.diffsuion = GaussianDiffusion(mean_type="eps")
        self.unet = UNetModel(
            in_channels=4,
            model_channels=64,
            out_channels=4,
            channel_mult=(1,),
            attention_resolutions=(),
            num_res_blocks=2,
        ).to(self.device)
        self.save_path = os.path.join(get_project_path(project_name="Defense"), "pretrained", "latent_diffusion64.pth")
        self.unet.load_state_dict(torch.load(self.save_path))

        self.ae_model = MaskMemoryAutoencoder(
            in_ch=1,
            ch=128,
            z_ch=4,
            out_ch=1,
            embed_dim=4,
            ch_mult=(1, 2, 4),
            resolution=28,
            num_res_blocks=2,
            attn_resolutions=(),
            dropout=0.0,
            double_z=False,
            num_memories=600,
            num_classes=10,
        ).to(self.device)
        self.ae_path = os.path.join(get_project_path("Defense"), "LDM", "models", "MaskMemoryAE_MNIST.pth")
        self.ae_model.load_state_dict(torch.load(self.ae_path))

        self.trainloader = trainloader
        self.testloader = testloader

    def train(self, epochs):
        start = time()
        optimizer = optim.AdamW(self.unet.parameters(), lr=self.lr, weight_decay=0.005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)
        self.ae_model.eval()
        for epoch in range(epochs):
            count = len(self.trainloader)
            epoch_loss = 0
            self.unet.train()
            for step, (img, _) in tqdm(enumerate(self.trainloader), desc=f"train step {epoch+1}/{epochs}", total=count):
                optimizer.zero_grad()
                img = img.to(self.device)
                batch_size = img.shape[0]

                t = self.diffsuion.get_rand_t(batch_size, device=self.device, min_t=0, max_t=1000)
                z = self.ae_model.encode(img)

                loss = self.diffsuion.training_losses(model=self.unet, x_start=z, t=t)
                epoch_loss += loss
                loss.backward()
                optimizer.step()

            if optimizer.state_dict()['param_groups'][0]['lr'] > self.lr / 1e2:
                scheduler.step()

            torch.save(self.unet.state_dict(), self.save_path)

            print(f"Epoch:{epoch+1}/{epochs}  Loss:{epoch_loss / count:.8f}")

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
        z = self.ae_model.encode(imgs)
        # torch.manual_seed(10)
        noise = torch.randn_like(z).to(self.device)
        # label = torch.randint(0, 10, (z.shape[0], ), device=self.device)
        label = [8] * z.shape[0]
        label = torch.LongTensor(label).to(self.device)
        generated_z = spacediffusion.ddim_sample_loop(self.unet, shape=z.shape, noise=noise, progress=True)

        generated_sample = self.ae_model.decode(self.ae_model.encode_encode(generated_z, label))[0]
        draw_img_groups([imgs, generated_sample])

    def test_ddpm(self):
        self.unet.eval()
        imgs, _ = next(iter(self.testloader))
        imgs = imgs.to(self.device)
        z = self.ae_model.encode(imgs)
        label = torch.randint(0, 10, (z.shape[0],), device=self.device)

        noise = torch.randn_like(z).to(self.device)
        generated_z = self.diffsuion.p_sample_loop(self.unet, shape=z.shape, noise=noise, progress=True)
        generated_sample = self.ae_model.decode(self.ae_model.encode_encode(generated_z, label))[0]
        draw_img_groups([imgs, generated_sample])


def main():
    trainer = TrainDiffusion()
    # trainer.train(5000)
    trainer.test_ddim()
    # trainer.test_ddpm()


if __name__ == "__main__":
    main()



