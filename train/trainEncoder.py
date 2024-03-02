from torch import optim, nn
from ContrastExperiances.Diffusion.models import UNetModel
from ContrastExperiances.Diffusion.respace import SpacedDiffusion
from Diffusion.encoder import Resnet34Encoder
from data.TJ_PV600 import trainloader32, testloader32
from time import time
from utils import *


class Trainer:
    def __init__(
            self,
            lr=1e-4,
    ):
        super(Trainer, self).__init__()
        self.device = "cuda"
        self.lr = lr

        self.model = Resnet34Encoder(image_size=32).to(self.device)
        self.save_path = os.path.join(get_project_path(project_name="Defense"), "pretrained", "encoder_diffusion32.pth")
        self.model.load_state_dict(torch.load(self.save_path))

        self.spacediffusion = SpacedDiffusion(num_ddpm_timesteps=1000, num_ddim_timesteps=50)
        self.unet = UNetModel(
            in_channels=1,
            model_channels=64,
            out_channels=1,
            channel_mult=(1, 2, 3, 4),
            num_res_blocks=2,
        ).to(self.device)
        diff_path = os.path.join(get_project_path(project_name="Defense"), "pretrained", "diffusion32.pth")
        self.unet.load_state_dict(torch.load(diff_path))

        self.trainloader = trainloader32
        self.testloader = testloader32

    def train(self, epochs):
        start = time()
        loss_fn = nn.MSELoss()

        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)

        self.unet.eval()
        for epoch in range(epochs):
            count = len(self.trainloader)
            epoch_loss = 0
            self.model.train()
            for step, (img, _) in tqdm(enumerate(self.trainloader), desc=f"train step {epoch+1}/{epochs}", total=count):
                optimizer.zero_grad()
                img = img.to(self.device)

                z = self.model(img).view(img.size(0), img.size(1), img.size(2), img.size(3))
                img_rec = self.spacediffusion.ddim_sample_loop(
                    model=self.unet, noise=z, shape=z.shape, device=self.device
                )
                loss = loss_fn(img, img_rec)
                loss.requires_grad_(True)
                loss.backward()
                optimizer.step()

                epoch_loss += loss

            print(f"Epoch:{epoch + 1}/{epochs}  Loss:{epoch_loss / count:.8f}")
            if optimizer.state_dict()['param_groups'][0]['lr'] > self.lr / 1e2:
                scheduler.step()

            torch.save(self.model.state_dict(), self.save_path)

            if (epoch+1) % 10 == 0 and epoch != 0:
                self.test()

        end = time()
        print_time_consumed(int(end - start))

    def test(self):
        self.model.eval()
        imgs, _ = next(iter(self.testloader))
        imgs = imgs.to(self.device)
        z = self.model(imgs).view(imgs.size(0), imgs.size(1), imgs.size(2), imgs.size(3))
        imgs_rec = self.spacediffusion.ddim_sample_loop(model=self.unet, noise=z, shape=z.shape, device=self.device)
        draw_img_groups([imgs, imgs_rec])


def main():
    trainer = Trainer()
    # trainer.train(200)
    trainer.test()


if __name__ == "__main__":
    main()



