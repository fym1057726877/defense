import os
import torch
from torch import optim
from ours.diffusion.models import MultilevelMemoryUnet
from data.TJ_PV600 import trainloader, testloader
from tqdm import tqdm
from time import time
from utils import *


class Trainer:
    def __init__(
            self,
            lr=2e-4,
    ):
        super(Trainer, self).__init__()
        self.device = "cuda"
        self.lr = lr
        self.unet = MultilevelMemoryUnet(
            in_channels=1,
            model_channels=32,
            out_channels=1,
            channel_mult=(1, 1, 2, 2, 2),
            num_res_blocks=2,
            features=1024,
            num_memories=600,
            threshold=1e-4,
        ).to(self.device)
        self.save_path = os.path.join(
            get_project_path(project_name="Defense"), "pretrained", "Multi_Memory_Unet_AE.pth")
        self.unet.load_state_dict(torch.load(self.save_path))

        self.trainloader = trainloader
        self.testloader = testloader

    def train(self, epochs):
        start = time()
        loss_fn = torch.nn.MSELoss()
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
                t = torch.LongTensor([0] * batch_size).to(self.device)
                rec, mem_loss = self.unet(img, t)
                loss = loss_fn(rec, img) + mem_loss

                epoch_loss += loss
                loss.backward()
                optimizer.step()

            if optimizer.state_dict()['param_groups'][0]['lr'] > self.lr / 1e2:
                scheduler.step()

            torch.save(self.unet.state_dict(), self.save_path)

            print(f"Epoch:{epoch+1}/{epochs}  Loss:{epoch_loss:.8f}")

            if epoch % 10 == 0 and epoch != 0:
                self.test()

        end = time()
        seconds = int(end - start)
        minutes = seconds // 60
        remain_second = seconds % 60
        print(f"time consumed: {minutes}min{remain_second}s")

    def test(self):
        self.unet.eval()
        imgs, _ = next(iter(self.testloader))
        imgs = imgs.to(self.device)
        t = torch.LongTensor([0] * imgs.shape[0]).to(self.device)
        rec, _ = self.unet(imgs, t)

        draw_img_groups([imgs, rec])


def main():
    # trainer = TrainDiffusion(unet_pred=ModelMeanType.START_X)
    trainer = Trainer()
    # trainer.train(2000)
    trainer.test()
if __name__ == "__main__":
    main()



