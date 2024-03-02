import os
import torch
import logging
from torch import optim
from data.TJ_PV600 import trainloader, testloader
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
            num_memories=600,
            features_channels=4,
            memory_threshold=1e-4,
            sparse=True,
            resolution=64
        ).to(self.device)
        self.save_path = os.path.join(
            get_project_path(project_name="Defense"), "ours", "memory_v2", "PV600", "memory_v2.pth")
        self.model.load_state_dict(torch.load(self.save_path))

        self.trainloader = trainloader
        self.testloader = testloader

        logging.basicConfig(filename="logs/memory_v2_train.log", filemode="w", level=logging.INFO, format="")

    def train(self, epochs):
        start = time()

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

            torch.save(self.model.state_dict(), self.save_path)

            print(f"Epoch:{epoch+1}/{epochs}  Loss:{epoch_loss:.8f}")

            if epoch != 0 and (epoch+1) % 5 == 0:
                accs = simple_test()
                logging.info(f"Epoch:{epoch+1}/{epochs}  "
                             f"nor_acc:{accs[0]:.3f}  "
                             f"rec_acc:{accs[1]:.3f}  "
                             f"adv_acc:{accs[2]:.3f}  "
                             f"rav_acc:{accs[3]:.3f}")

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
    # trainer.train(2000)
    # trainer.test()
    trainer.test_defense()


if __name__ == "__main__":
    main()

