import os
import torch
import torch.nn as nn
import logging
from torch import optim
from data.TJ_PV600 import trainloader, testloader
from ContrastExperiances.TransGan.Actor import Resnet34Actor
from ContrastExperiances.TransGan.TransGanModel import SwinTransGenerator
from utils import get_project_path, draw_img_groups
from tqdm import tqdm
from ContrastExperiances.TransGan.testTransGan import simple_test

# dataset_name = "Handvein"
# dataset_name = "Handvein3"
dataset_name = "Fingervein2"

# is_local = True
is_local = False
is_peg = True
# is_peg = False

width = 64
height = 64
bottom_width = 8
bottom_height = 8
window_size = 4

g_embed_dim = 512
d_embed_dim = 768

# g_depths = [2, 4, 2, 2]
g_depths = [4, 2, 2, 2]
# d_depths = [2, 2, 6, 2]
d_depths = [4, 2, 2, 2]


class Trainer:
    def __init__(self):
        super().__init__()

        self.device = "cuda"
        self.lr = 2e-4
        self.actor = Resnet34Actor().to(self.device)
        self.actor_path = os.path.join(
            get_project_path(project_name="Defense"),
            "ContrastExperiances",
            "TransGan",
            "resnet34actor.pth"
        )
        self.generator = SwinTransGenerator(
            embed_dim=g_embed_dim,
            bottom_width=bottom_width,
            bottom_height=bottom_height,
            window_size=window_size,
            depth=g_depths,
            is_local=is_local,
            is_peg=is_peg
        ).to(self.device)
        self.gen_path = os.path.join(
            get_project_path(project_name="Defense"),
            "ContrastExperiances",
            "TransGan",
            "checkpoint",
            "TransGenerator.pth"
        )
        self.actor.load_state_dict(torch.load(self.actor_path))
        self.generator.load_state_dict(torch.load(self.gen_path))

        logging.basicConfig(filename="logs/actor_train.log", filemode="w", level=logging.INFO, format="")

    def train(self, epochs):
        self.test_defense()
        optimizer_actor = optim.Adam(self.actor.parameters(), self.lr)
        lr_scheduler_actor = optim.lr_scheduler.StepLR(optimizer_actor, step_size=1, gamma=0.99)
        mse_loss_fun = nn.MSELoss()
        count = len(trainloader)
        self.generator.eval()
        for epoch in range(epochs):
            self.actor.train()
            epoch_loss = 0
            for index, (img, label) in tqdm(enumerate(trainloader), total=count, desc=f"train step{epoch+1}/{epochs}"):
                img = img.to(self.device)
                optimizer_actor.zero_grad()
                z = self.actor(img)
                recImg = self.generator(z)
                train_loss = mse_loss_fun(img, recImg)
                train_loss.backward()
                optimizer_actor.step()
                epoch_loss += train_loss
            if optimizer_actor.state_dict()["param_groups"][0]["lr"] > self.lr / 1e2:
                lr_scheduler_actor.step()

            print(f"epoch{epoch}/{epochs}   loss:{epoch_loss/count:.8f}")
            torch.save(self.actor.state_dict(), self.actor_path)

            if epoch != 0 and (epoch + 1) % 5 == 0:
                accs = simple_test()
                logging.info(f"Epoch:{epoch + 1}/{epochs}  "
                             f"nor_acc:{accs[0]:.3f}  "
                             f"rec_acc:{accs[1]:.3f}  "
                             f"adv_acc:{accs[2]:.3f}  "
                             f"rav_acc:{accs[3]:.3f}")

    def test(self):
        self.actor.eval()
        imgs, _ = next(iter(testloader))
        imgs = imgs.to(self.device)
        z = self.actor(imgs)
        rec = self.generator(z)
        draw_img_groups([imgs, rec])

    def test_defense(self):
        simple_test()


def trainDirectActor():
    # seed = 1  # seed for random function
    directActorRec = Trainer()
    directActorRec.train(2000)
    # directActorRec.test_defense()


if __name__ == "__main__":
    trainDirectActor()
