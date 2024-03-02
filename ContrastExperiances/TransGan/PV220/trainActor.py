import os
import torch
import torch.nn as nn
import logging
from torch import optim
from data.VERA_PV220 import trainloader, testloader
from ContrastExperiances.TransGan.Actor import Resnet34Actor
from ContrastExperiances.TransGan.TransGanModel import SwinTransGenerator
from utils import get_project_path, draw_img_groups, simple_test_defense
from tqdm import tqdm

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
        self.model = Resnet34Actor().to(self.device)
        self.save_path = os.path.join(
            get_project_path(project_name="Defense"),
            "ContrastExperiances",
            "TransGan",
            "PV220",
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
            "PV220",
            "TransGenerator.pth"
        )
        # self.model.load_state_dict(torch.load(self.actor_path))
        self.generator.load_state_dict(torch.load(self.gen_path))

        self.trainloader = trainloader
        self.testloader = testloader
        self.classifier_name = "ModelB_PV220"

    def train(self, epochs):
        logging.basicConfig(filename="logs/actor_train.log", filemode="w", level=logging.INFO, format="")
        best_acc = 0
        optimizer_actor = optim.Adam(self.model.parameters(), self.lr)
        lr_scheduler_actor = optim.lr_scheduler.StepLR(optimizer_actor, step_size=1, gamma=0.99)
        mse_loss_fun = nn.MSELoss()
        count = len(trainloader)
        self.generator.eval()
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            for index, (img, label) in tqdm(enumerate(trainloader), total=count, desc=f"train step{epoch+1}/{epochs}"):
                img = img.to(self.device)
                optimizer_actor.zero_grad()
                z = self.model(img)
                recImg = self.generator(z)
                train_loss = mse_loss_fun(img, recImg)
                train_loss.backward()
                optimizer_actor.step()
                epoch_loss += train_loss
            if optimizer_actor.state_dict()["param_groups"][0]["lr"] > self.lr / 1e2:
                lr_scheduler_actor.step()

            log = simple_test_defense(defense_model=self.model, defense_fn=self.defense, attack_type="FGSM", eps=0.03,
                                      progress=False, classifier_name=self.classifier_name, testloader=self.testloader,
                                      print_info=True if epoch == 0 else False)
            if log["RAvAcc"] > best_acc:
                print("saving the best model")
                torch.save(self.model.state_dict(), self.save_path)
                best_acc = log["RAvAcc"]
            log["best_acc"] = best_acc
            log = {"epoch": epoch + 1, "loss": f"{epoch_loss:.8f}", **log}
            print(log)
            logging.info(log)

    def defense(self, x, model):
        z = model(x)
        rec = self.generator(z)
        return rec

    def test(self):
        self.model.eval()
        imgs, _ = next(iter(testloader))
        imgs = imgs.to(self.device)
        z = self.model(imgs)
        rec = self.generator(z)
        draw_img_groups([imgs, rec])


def trainDirectActor():
    # seed = 1  # seed for random function
    directActorRec = Trainer()
    directActorRec.train(2000)
    # directActorRec.test_defense()


if __name__ == "__main__":
    trainDirectActor()
