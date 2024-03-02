import os
import torch
import logging
from torch import optim
from data.FV500 import trainloader, testloader
from tqdm import tqdm
from time import time
from utils import get_project_path, simple_test_defense, print_time_consumed, draw_img_groups
from ours.memory_v2.models import MemoryAE_v2


class Trainer:
    def __init__(
            self,
            lr=5e-4,
    ):
        super(Trainer, self).__init__()
        self.device = "cuda"
        self.lr = lr

        self.model = MemoryAE_v2(
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
            get_project_path(project_name="Defense"), "ours", "memory_v2", "PV500", "memory_v2_PV500.pth")
        self.model.load_state_dict(torch.load(self.save_path))

        self.trainloader = trainloader
        self.testloader = testloader

        logging.basicConfig(filename="./logs/memory_v2_PV500_train.log", filemode="w", level=logging.INFO, format="")
        self.classifier_name = "Vit_FV500"

    def train(self, epochs):
        start = time()
        best_acc = 0
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)

        for epoch in range(epochs):
            count = len(self.trainloader)
            epoch_loss = 0
            self.model.train()
            for step, (img, _) in tqdm(enumerate(self.trainloader), desc=f"train step {epoch+1}/{epochs}", total=count):
                optimizer.zero_grad()
                img = img.to(self.device)

                loss = self.model.training_loss(img)

                loss.backward()
                optimizer.step()

                epoch_loss += loss

            if optimizer.state_dict()['param_groups'][0]['lr'] > self.lr / 1e2:
                scheduler.step()

            # torch.save(self.model.state_dict(), self.save_path)

            print(f"Epoch:{epoch+1}/{epochs}  Loss:{epoch_loss:.8f}")

            with torch.no_grad():
                epoch_loss = epoch_loss / count
                log = simple_test_defense(defense_model=self.model, attack_type="FGSM", eps=0.1, progress=False,
                                          classifier_name=self.classifier_name, testloader=self.testloader,
                                          print_info=True if epoch == 0 else False)
                if log["RAvAcc"] > best_acc:
                    print("saving the best model")
                    torch.save(self.model.state_dict(), self.save_path)
                    best_acc = log["RAvAcc"]
                log["best_acc"] = best_acc
                log = {"epoch": epoch + 1, "loss": f"{epoch_loss:.8f}", **log}
                print(log)
                logging.info(log)

        end = time()
        seconds = int(end - start)
        print_time_consumed(seconds)

    def test_defense(self):
        classifier_name = "Vit_FV500"
        log = simple_test_defense(defense_model=self.model, attack_type="FGSM", eps=0.1,
                                  testloader=self.testloader, progress=True,
                                  classifier_name=classifier_name, device=self.device)
        print(log)


def main():
    trainer = Trainer()
    # trainer.train(2000)
    # trainer.test()
    trainer.test_defense()


if __name__ == "__main__":
    main()

