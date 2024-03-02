import os
import torch
from tqdm import tqdm
from time import time
from utils import get_project_path, print_time_consumed, draw_img_groups
from LDM.dataset.mnist import trainloader, testloader
from LDM.aemodel.autoencoder import MaskMemoryAutoencoder


class Trainer:
    def __init__(self):
        super().__init__()
        self.device = "cuda"
        self.model = MaskMemoryAutoencoder(
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
        self.save_path = os.path.join(get_project_path("Defense"), "LDM", "models", "MaskMemoryAE_MNIST.pth")
        self.model.load_state_dict(torch.load(self.save_path))

        self.trainloader = trainloader
        self.testloader = testloader

    def train(self, epochs):
        start = time()

        optimizer_ae, optimizer_disc = self.model.config_optimizer(lr=1e-4)

        for epoch in range(epochs):
            count = len(self.trainloader)
            epoch_loss_ae, epoch_loss_disc = 0, 0
            self.model.train()
            for step, (img, label) in tqdm(enumerate(self.trainloader), desc=f"train step {epoch + 1}/{epochs}",
                                           total=count):
                # train ae
                optimizer_ae.zero_grad()
                img, label = img.to(self.device), label.to(self.device)
                ae_loss = self.model.training_loss(input=img, label=label, optimizer_idx=0)
                ae_loss.backward()
                optimizer_ae.step()
                epoch_loss_ae += ae_loss

                # train disc
                optimizer_disc.zero_grad()
                disc_loss = self.model.training_loss(input=img, label=label, optimizer_idx=1)
                disc_loss.backward()
                optimizer_disc.step()
                epoch_loss_disc += disc_loss

            torch.save(self.model.state_dict(), self.save_path)
            with torch.no_grad():
                epoch_loss_ae = round(float(epoch_loss_ae / count), 8)
                epoch_loss_disc = round(float(epoch_loss_disc / count), 8)
                log = {"epoch": epoch + 1,
                       "ae_loss": f"{epoch_loss_ae:.8f}",
                       "disc_loss": f"{epoch_loss_disc:.8f}"}
                print(log)

                if (epoch+1) % 1 == 0:
                    self.test_rec()

        end = time()
        seconds = int(end - start)
        print_time_consumed(seconds)

    def test_rec(self):
        self.model.eval()
        imgs, label = next(iter(self.testloader))
        imgs, label = imgs.to(self.device), label.to(self.device)
        imgs_rec = self.model(imgs, label)[0]
        draw_img_groups([imgs, imgs_rec])


def main():
    trainer = Trainer()
    trainer.train(2000)
    # trainer.test_rec()


if __name__ == "__main__":
    main()
