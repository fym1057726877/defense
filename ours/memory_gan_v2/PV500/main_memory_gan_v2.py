import os
import torch
import logging
from tqdm import tqdm
from time import time
from utils import get_project_path, print_time_consumed, draw_img_groups, simple_test_defense
from data.FV500 import trainloader, testloader
from ours.models import MemoryGAN_v2


class Trainer:
    def __init__(
            self,
    ):
        super().__init__()
        self.device = "cuda"
        self.model = MemoryGAN_v2(
            in_channel=1,
            channel=256,
            n_res_block=2,
            n_res_channel=64,
            num_memories=1000,
            features_channels=4,
            memory_threshold=None,
            sparse=True,
            resolution=64
        ).to(self.device)
        self.save_path = os.path.join(get_project_path("Defense"), "ours", "memory_gan_v2", "PV500", "memory_gan_v2_PV500.pth")
        self.model.load_state_dict(torch.load(self.save_path))
        self.classifier_name = "Vit_FV500"
        self.trainloader = trainloader
        self.testloader = testloader

    def train(self, epochs):
        start = time()

        optimizer_ae, optimizer_disc = self.model.config_optimizer(lr=1e-4)
        best_acc = 0

        for epoch in range(epochs):
            count = len(self.trainloader)
            epoch_loss_ae, epoch_loss_disc = 0, 0
            self.model.train()
            for step, (img, _) in enumerate(self.trainloader):
                # train ae
                optimizer_ae.zero_grad()
                img = img.to(self.device)
                ae_loss = self.model.training_loss(img, optimizer_idx=0)
                ae_loss.backward()
                optimizer_ae.step()
                epoch_loss_ae += ae_loss

                # train disc
                optimizer_disc.zero_grad()
                disc_loss = self.model.training_loss(img, optimizer_idx=1)
                disc_loss.backward()
                optimizer_disc.step()
                epoch_loss_disc += disc_loss

            with torch.no_grad():
                epoch_loss_ae = round(float(epoch_loss_ae / count), 8)
                epoch_loss_disc = round(float(epoch_loss_disc / count), 8)
                log = simple_test_defense(defense_model=self.model, attack_type="FGSM", eps=0.1,
                                          classifier_name=self.classifier_name,
                                          print_info=True if epoch == 0 else False)
                if log["RAvAcc"] > best_acc:
                    print("saving the best model")
                    torch.save(self.model.state_dict(), self.save_path)
                    best_acc = log["RAvAcc"]
                log["best_acc"] = best_acc
                log = {"epoch": epoch + 1,
                       "ae_loss": f"{epoch_loss_ae:.8f}",
                       "epoch_loss_disc": f"{epoch_loss_disc}",
                       **log}
                print(log)
                logging.info(log)

        end = time()
        seconds = int(end - start)
        print_time_consumed(seconds)


def main():
    trainer = Trainer()
    trainer.train(2000)
    # trainer.test_defense()
    # trainer.total_test()
    # trainer.test_defense_one()


if __name__ == "__main__":
    main()
