import os
import torch
import logging
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from ContrastExperiances.MagNet.defensive_models import Magnet
from utils import get_project_path, draw_img_groups, simple_test_defense, get_time_cost
from data.TJ_PV600 import trainloader, testloader
from classifier.models import load_ModelB_PV600


class Trainer:
    def __init__(
            self,
    ):
        super(Trainer, self).__init__()
        self.device = "cuda"
        self.trainloader, self.testloader = trainloader, testloader
        self.weight_noise = 0.01

        # magnet
        self.model = Magnet().to(self.device)
        self.save_path = os.path.join(
            get_project_path("Defense"), "ContrastExperiances", "MagNet", "PV600", "magnet_pv600.pth")
        self.model.load_state_dict((torch.load(self.save_path)))

        # loss function
        self.loss_fun = nn.MSELoss()

        # optimizer
        self.optimer = optim.Adam(self.model.parameters(), lr=5e-4)

    def train(self, epochs):
        for e in range(epochs):
            train_num = 0
            epoch_loss = 0
            self.model.train()
            batch_count = len(self.trainloader)
            for index, (img, label) in tqdm(enumerate(self.trainloader), desc=f"train {e+1}/{epochs}",
                                            total=batch_count):
                self.optimer.zero_grad()
                img_ori, label = img.to(self.device), label.to(self.device)
                # img_noise = img_ori + self.weight_noise * torch.randn_like(img, device=self.device)
                img_noise = img_ori

                rec_x = self.model(img_noise)
                loss = self.loss_fun(rec_x, img_ori)

                loss.backward()
                self.optimer.step()

                train_num += label.size(0)
                epoch_loss += loss

            epoch_loss /= batch_count
            print(f"[Epoch {e+1}/{epochs}   Loss:{epoch_loss:.6f}]")
            torch.save(self.model.state_dict(), self.save_path)

            self.eval_classifier()

    def eval_rec(self):
        self.model.eval()
        imgs, labels = next(iter(self.testloader))
        imgs = imgs.to(self.device)
        x_recon = self.model(imgs)
        draw_img_groups([imgs, x_recon])

    def eval_classifier(self):
        target_classifier = load_ModelB_PV600()

        def accurary(y_pred, y):
            return (y_pred.max(dim=1)[1] == y).sum()

        total = 0
        acc_ori, acc_rec = 0., 0.
        for i, (img, label) in tqdm(enumerate(self.testloader), total=len(testloader), desc="eval classifier"):
            total += img.shape[0]
            img, label = img.to(self.device), label.to(self.device)

            y_ori = target_classifier(img)
            acc_ori += accurary(y_ori, label)

            rec_x = self.model(img)
            y_rec = target_classifier(rec_x)
            acc_rec += accurary(y_rec, label)

        acc_ori /= total
        acc_rec /= total
        print(f"acc_ori:{acc_ori:.8f}\n"
              f"acc_rec:{acc_rec:.8f}")

    def total_test1(self):
        torch.manual_seed(10)
        classifier_name_list = [
            "Res2Net_PV600",
            "Vit_PV600",
            "SwinTransformer_PV600",
        ]
        for classifier_name in classifier_name_list:
            log1 = simple_test_defense(defense_model=self.model, attack_type="FGSM", eps=0.01,
                                       testloader=self.testloader, progress=False,
                                       classifier_name=classifier_name, device=self.device)
            print(log1)
            log2 = simple_test_defense(defense_model=self.model, attack_type="FGSM", eps=0.015,
                                       testloader=self.testloader, progress=False,
                                       classifier_name=classifier_name, device=self.device)
            print(log2)
            log3 = simple_test_defense(defense_model=self.model, attack_type="PGD", eps=0.01,
                                       testloader=self.testloader, progress=False,
                                       classifier_name=classifier_name, device=self.device)
            print(log3)
            log4 = simple_test_defense(defense_model=self.model, attack_type="PGD", eps=0.015,
                                       testloader=self.testloader, progress=False,
                                       classifier_name=classifier_name, device=self.device)
            print(log4)

    def total_test(self):
        torch.manual_seed(10)
        logging.basicConfig(filename="./logs/magnet_pv600_white.logs", filemode="w", level=logging.INFO, format="")
        classifier_name_list = [
            "FV_CNN_PV600",
            "PV_CNN_PV600",
            "LightweightFVCNN_PV600",
            "FVRAS_Net_PV600",
            "ModelB_PV600",
        ]
        for classifier_name in classifier_name_list:
            log1 = simple_test_defense(defense_model=self.model, attack_type="FGSM", eps=0.03,
                                       testloader=self.testloader, progress=False,
                                       classifier_name=classifier_name, device=self.device)
            print(log1)
            logging.info(log1)
            log2 = simple_test_defense(defense_model=self.model, attack_type="FGSM", eps=0.05,
                                       testloader=self.testloader, progress=False,
                                       classifier_name=classifier_name, device=self.device)
            print(log2)
            logging.info(log2)
            log3 = simple_test_defense(defense_model=self.model, attack_type="PGD", eps=0.03,
                                       testloader=self.testloader, progress=False,
                                       classifier_name=classifier_name, device=self.device)
            print(log3)
            logging.info(log3)
            log4 = simple_test_defense(defense_model=self.model, attack_type="PGD", eps=0.05,
                                       testloader=self.testloader, progress=False,
                                       classifier_name=classifier_name, device=self.device)
            print(log4)
            logging.info(log4)

    def BlackBoxAttackTest(self):
        torch.manual_seed(10)
        logging.basicConfig(filename="logs/magnet_pv600_test_black.logs", filemode="w", level=logging.INFO, format="")
        classifier_name_list = [
            # "FV_CNN_PV600",
            "PV_CNN_PV600",
            # "LightweightFVCNN_PV600",
            "FVRAS_Net_PV600",
            # "ModelB_PV600",
            # "Vit_PV600",
            # "Res2Net_PV600",
            # "SwinTransformer_PV600",
        ]
        for classifier_name in classifier_name_list:
            log1 = simple_test_defense(defense_model=self.model, attack_type="SPSA", eps=0.2,
                                       testloader=self.testloader, progress=True,
                                       classifier_name=classifier_name, device=self.device)
            print(log1)
            logging.info(log1)

    def test_time_cost(self):
        classifier_name = "PV_CNN_PV600"
        time_cost = get_time_cost(defense_model=self.model, classifier_name=classifier_name,
                                  attack_type="FGSM", eps=0.03, progress=False, device=self.device)
        print(time_cost)


def main():
    trainer = Trainer()
    # trainer.train(500)
    # trainer.total_test1()
    # trainer.BlackBoxAttackTest()
    trainer.test_time_cost()

if __name__ == "__main__":
    main()
