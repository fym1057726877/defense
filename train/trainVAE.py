
import torch
from torch import optim
from data.TJ_PV600 import trainloader, testloader
from tqdm import tqdm
from time import time
from utils import *
from LDM.aemodel.autoencoder import AutoencoderKL


class Trainer:
    def __init__(
            self,
            lr=5e-5,
    ):
        super(Trainer, self).__init__()
        self.device = "cuda"
        self.lr = lr

        self.model = AutoencoderKL(
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
            double_z=True
        ).to(self.device)

        self.save_path = os.path.join(get_project_path(project_name="Defense"), "pretrained", "VAE.pth")
        self.model.load_state_dict(torch.load(self.save_path), strict=False)

        self.trainloader = trainloader
        self.testloader = testloader

    def train(self, epochs):
        start = time()

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.9))

        for epoch in range(epochs):
            count = len(self.trainloader)
            self.model.train()
            ae_epoch_loss = 0
            for step, (img, _) in tqdm(enumerate(self.trainloader), desc=f"train step {epoch+1}/{epochs}", total=count):
                optimizer.zero_grad()
                img = img.to(self.device)

                loss = self.model.training_loss(img)
                loss.backward()
                optimizer.step()

                ae_epoch_loss += loss.item()

            ae_epoch_loss /= count

            print(f"Epoch:{epoch+1}/{epochs}  Loss:{ae_epoch_loss:.8f}")

            if epoch != 0 and (epoch+1) % 3 == 0:
                self.test()

            torch.save(self.model.state_dict(), self.save_path)

        end = time()
        seconds = int(end - start)
        print_time_consumed(seconds)

    def test(self):
        self.model.eval()
        imgs, _ = next(iter(self.testloader))
        imgs = imgs.to(self.device)
        imgs_rec, _ = self.model(imgs)
        draw_img_groups([imgs, imgs_rec])


def main():
    trainer = Trainer()
    # trainer.train(100)
    trainer.test()


if __name__ == "__main__":
    main()

