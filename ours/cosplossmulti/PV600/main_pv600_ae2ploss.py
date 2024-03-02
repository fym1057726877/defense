import os
import torch
import logging
from data.TJ_PV600 import trainloader, testloader
from tqdm import tqdm
from time import time
from utils import get_project_path, simple_test_defense, print_time_consumed, draw_img_groups
from ours.cosplossmulti.models import MemoryAE_v2_ploss


class Trainer:
    def __init__(
            self,
            lr=5e-4,
    ):
        super(Trainer, self).__init__()
        self.device = "cuda"
        self.lr = lr

        self.model = MemoryAE_v2_ploss(
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
        self.save_path = os.path.join(
            get_project_path(project_name="Defense"), "ours", "cosplossmulti", "PV600", "memory_v2_pv600.pth")
        self.model.load_state_dict(torch.load(self.save_path))

        self.trainloader = trainloader
        self.testloader = testloader

        self.classifier_name = "Vit_PV600"

    def train(self, epochs, progress=False):
        start = time()

        logging.basicConfig(filename="./logs/memoryae_v2_ploss_train.logs", filemode="w", level=logging.INFO, format="")
        optimizer_ae = self.model.config_optimizer(lr=2e-4)
        best_acc = 0

        for epoch in range(epochs):
            count = len(self.trainloader)
            epoch_loss_ae = 0
            self.model.train()
            if progress:
                iter_object = tqdm(enumerate(self.trainloader), desc=f"train step {epoch + 1}/{epochs}", total=count)
            else:
                iter_object = enumerate(self.trainloader)
            for step, (img, _) in iter_object:
                optimizer_ae.zero_grad()
                img = img.to(self.device)
                ae_loss = self.model.training_loss(img)
                ae_loss.backward()
                optimizer_ae.step()
                epoch_loss_ae += ae_loss

            # torch.save(self.model.state_dict(), self.save_path)
            with torch.no_grad():
                epoch_loss_ae = round(float(epoch_loss_ae / count), 8)
                log = simple_test_defense(defense_model=self.model, attack_type="FGSM", eps=0.01,
                                          classifier_name=self.classifier_name, progress=False,
                                          print_info=True if epoch == 0 else False)
                if log["RAvAcc"] > best_acc:
                    print("saving the best model")
                    torch.save(self.model.state_dict(), self.save_path)
                    best_acc = log["RAvAcc"]
                log["best_acc"] = best_acc
                log = {"epoch": epoch + 1,
                       "ae_loss": f"{epoch_loss_ae:.8f}",
                       **log}
                print(log)
                logging.info(log)

        end = time()
        seconds = int(end - start)
        print_time_consumed(seconds)

    def test_defense(self):
        classifier_name = "Vit_PV600"
        log = simple_test_defense(defense_model=self.model, attack_type="FGSM", eps=0.01,
                                  testloader=self.testloader, progress=True,
                                  classifier_name=classifier_name, device=self.device)
        print(log)


def main():
    trainer = Trainer()
    trainer.train(2000)
    # trainer.test()
    # trainer.test_defense()


if __name__ == "__main__":
    main()

