import os
import torch
from torch import optim, nn
from data.TJ_PV600 import trainloader, testloader
from tqdm import tqdm
from time import time
from utils import *
from LDM.aemodel.autoencoder import MultiLevelMemoryAutoencoder


class Trainer:
    def __init__(
            self,
            lr=5e-4,
    ):
        super(Trainer, self).__init__()
        self.device = "cuda"
        self.lr = lr

        self.model = MultiLevelMemoryAutoencoder(in_ch=1, ch=32, z_ch=4, out_ch=1, embed_dim=4, ch_mult=(1, 2, 4),
                                                 resolution=64, num_res_blocks=2, mem_resolutions=(16,), dropout=0.0,
                                                 num_memories=200, threshold=1e-5, recloss_type="l1", sparse=True,
                                                 ).to(self.device)
        self.save_path = os.path.join(get_project_path(project_name="Defense"), "pretrained", "MLMAE.pth")
        # self.model.load_state_dict(torch.load(self.save_path))

        self.trainloader = trainloader
        self.testloader = testloader

    def train(self, epochs):
        start = time()

        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)

        for epoch in range(epochs):
            count = len(self.trainloader)
            epoch_loss = 0
            self.model.train()
            for step, (img, _) in tqdm(enumerate(self.trainloader), desc=f"train step {epoch + 1}/{epochs}",
                                       total=count):
                optimizer.zero_grad()
                img = img.to(self.device)

                loss = self.model.training_loss(img)

                loss.backward()
                optimizer.step()

                epoch_loss += loss

            if optimizer.state_dict()['param_groups'][0]['lr'] > self.lr / 1e2:
                scheduler.step()

            torch.save(self.model.state_dict(), self.save_path)

            print(f"Epoch:{epoch + 1}/{epochs}  Loss:{epoch_loss / count:.8f}")

            if epoch != 0 and (epoch + 1) % 10 == 0:
                self.test()

        end = time()
        seconds = int(end - start)
        print_time_consumed(seconds)

    def test(self):
        self.model.eval()
        imgs, _ = next(iter(self.testloader))
        imgs = imgs.to(self.device)
        imgs_rec = self.model(imgs)[0]
        draw_img_groups([imgs, imgs_rec])


def main():
    trainer = Trainer()
    trainer.train(500)
    # trainer.test()


if __name__ == "__main__":
    main()
