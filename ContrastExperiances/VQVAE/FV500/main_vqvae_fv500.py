import os
import torch
import logging
from tqdm import tqdm
from time import time
from data.FV500 import trainloader, testloader
from ContrastExperiances.VQVAE.models import VQVAE
from utils import get_project_path, print_time_consumed, draw_img_groups, simple_test_defense, get_time_cost


class Trainer:
    def __init__(
            self,
            lr=1e-4,
    ):
        super(Trainer, self).__init__()
        self.device = "cuda"
        self.lr = lr

        self.model = VQVAE(
            in_channel=1,
            channel=256,
            n_res_block=2,
            n_res_channel=64,
            embed_dim=64,
            n_embed=512,
        ).to(self.device)
        self.save_path = os.path.join(
            get_project_path(project_name="Defense"), "ContrastExperiances", "VQVAE", "FV500", "VQVAE_FV500.pth")
        self.model.load_state_dict(torch.load(self.save_path))

        self.trainloader = trainloader
        self.testloader = testloader

        self.classifier_name = "ModelB_FV500"
        # self.classifier_name = "Vit_PV600"
        # self.classifier_name = "Res2Net_PV600"

    def train(self, epochs, progress=True):
        start = time()
        best_acc = 0
        logging.basicConfig(filename="./logs/vqvae_train_FV500.log", filemode="w", level=logging.INFO, format="")
        optimizer = self.model.config_optimizer(lr=self.lr)

        for epoch in range(epochs):
            count = len(self.trainloader)
            epoch_loss = 0
            self.model.train()
            if progress:
                iterobject = tqdm(enumerate(self.trainloader), desc=f"train step {epoch + 1}/{epochs}", total=count)
            else:
                iterobject = enumerate(self.trainloader)
            for step, (img, _) in iterobject:
                optimizer.zero_grad()
                img = img.to(self.device)
                loss = self.model.training_loss(img)
                loss.backward()
                optimizer.step()
                epoch_loss += loss

            with torch.no_grad():
                epoch_loss = epoch_loss / count
                log = simple_test_defense(defense_model=self.model, attack_type="FGSM", eps=0.03,
                                          classifier_name=self.classifier_name, progress=False,
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

    def test(self):
        self.model.eval()
        imgs, _ = next(iter(self.testloader))
        imgs = imgs.to(self.device)
        imgs_rec, _ = self.model(imgs)
        draw_img_groups([imgs, imgs_rec])

    def test_defense_one(self):
        # classifier_name = "ModelB_FV500"
        # classifier_name = "Vit_FV500"
        # classifier_name = "Res2Net_FV500"
        # classifier_name = "SwinTransformer_FV500"

        # classifier_name = "FV_CNN_FV500"
        classifier_name = "PV_CNN_FV500"
        # classifier_name = "LightweightFVCNN_FV500"
        # classifier_name = "FVRAS_Net_FV500"
        log = simple_test_defense(defense_model=self.model, attack_type="SPSA", eps=0.3,
                                  testloader=self.testloader,
                                  classifier_name=classifier_name, device=self.device)
        print(log)

        # classifier_name1 = "Vit_FV500"
        # log1 = simple_test_defense(defense_model=self.model, attack_type="SPSA", eps=0.3,
        #                            testloader=self.testloader,
        #                            classifier_name=classifier_name1, device=self.device)
        # print(log1)

    def test_defense(self):
        torch.manual_seed(10)
        # classifier_name = "ModelB_FV500"
        classifier_name = "Vit_FV500"
        # classifier_name = "Res2Net_FV500"
        # classifier_name = "SwinTransformer_FV500"

        # classifier_name = "FV_CNN_FV500"
        # classifier_name = "PV_CNN_FV500"
        # classifier_name = "LightweightFVCNN_FV500"
        # classifier_name = "FVRAS_Net_FV500"

        log1 = simple_test_defense(defense_model=self.model, attack_type="FGSM", eps=0.3,
                                   testloader=self.testloader, progress=False,
                                   classifier_name=classifier_name, device=self.device)
        print(log1)
        # log2 = simple_test_defense(defense_model=self.model, attack_type="FGSM", eps=0.015,
        #                            testloader=self.testloader, progress=False,
        #                            classifier_name=classifier_name, device=self.device)
        # print(log2)
        # log3 = simple_test_defense(defense_model=self.model, attack_type="PGD", eps=0.01,
        #                            testloader=self.testloader, progress=False,
        #                            classifier_name=classifier_name, device=self.device)
        # print(log3)
        log4 = simple_test_defense(defense_model=self.model, attack_type="PGD", eps=0.3,
                                   testloader=self.testloader, progress=False,
                                   classifier_name=classifier_name, device=self.device)
        print(log4)

    def total_test(self):
        torch.manual_seed(10)
        logging.basicConfig(filename="./logs/test1.logs", filemode="w", level=logging.INFO, format="")
        classifier_name_list = [
            "FV_CNN_FV500",
            "PV_CNN_FV500",
            "LightweightFVCNN_FV500",
            "FVRAS_Net_FV500",
            "ModelB_FV500",
        ]
        for classifier_name in classifier_name_list:
            log1 = simple_test_defense(defense_model=self.model, attack_type="FGSM", eps=0.3,
                                       testloader=self.testloader, progress=False,
                                       classifier_name=classifier_name, device=self.device)
            print(log1)
            logging.info(log1)
            # log2 = simple_test_defense(defense_model=self.model, attack_type="FGSM", eps=0.15,
            #                            testloader=self.testloader, progress=False,
            #                            classifier_name=classifier_name, device=self.device)
            # print(log2)
            # logging.info(log2)
            # log3 = simple_test_defense(defense_model=self.model, attack_type="PGD", eps=0.1,
            #                            testloader=self.testloader, progress=False,
            #                            classifier_name=classifier_name, device=self.device)
            # print(log3)
            # logging.info(log3)
            log4 = simple_test_defense(defense_model=self.model, attack_type="PGD", eps=0.3,
                                       testloader=self.testloader, progress=False,
                                       classifier_name=classifier_name, device=self.device)
            print(log4)
            logging.info(log4)

    def total_test1(self):
        torch.manual_seed(10)
        logging.basicConfig(filename="./logs/test1.logs", filemode="w", level=logging.INFO, format="")
        classifier_name_list = [
            "Res2Net_FV500",
            "Vit_FV500",
            "SwinTransformer_FV500",
        ]
        for classifier_name in classifier_name_list:
            log1 = simple_test_defense(defense_model=self.model, attack_type="FGSM", eps=0.1,
                                       testloader=self.testloader, progress=False,
                                       classifier_name=classifier_name, device=self.device)
            print(log1)
            logging.info(log1)
            log2 = simple_test_defense(defense_model=self.model, attack_type="FGSM", eps=0.2,
                                       testloader=self.testloader, progress=False,
                                       classifier_name=classifier_name, device=self.device)
            print(log2)
            logging.info(log2)
            log3 = simple_test_defense(defense_model=self.model, attack_type="PGD", eps=0.1,
                                       testloader=self.testloader, progress=False,
                                       classifier_name=classifier_name, device=self.device)
            print(log3)
            logging.info(log3)
            log4 = simple_test_defense(defense_model=self.model, attack_type="PGD", eps=0.2,
                                       testloader=self.testloader, progress=False,
                                       classifier_name=classifier_name, device=self.device)
            print(log4)
            logging.info(log4)

    def BlackBoxAttackTest(self):
        torch.manual_seed(10)
        logging.basicConfig(filename="./logs/blackbox.logs", filemode="w", level=logging.INFO, format="")
        classifier_name_list = [
            # "FV_CNN_FV500",
            "PV_CNN_FV500",
            # "LightweightFVCNN_FV500",
            "FVRAS_Net_FV500",
            "ModelB_FV500",
            # "Vit_FV500",
            # "Res2Net_FV500",
            "SwinTransformer_FV500",
        ]
        for classifier_name in classifier_name_list:
            log1 = simple_test_defense(defense_model=self.model, attack_type="Pixle", eps=3,
                                       testloader=self.testloader, progress=True, iters=128,
                                       classifier_name=classifier_name, device=self.device)
            print(log1)
            logging.info(log1)

    def test_time_cost(self):
        classifier_name = "PV_CNN_FV500"
        time_cost = get_time_cost(defense_model=self.model, classifier_name=classifier_name,
                                  attack_type="FGSM", eps=0.1, progress=False, device=self.device)
        print(time_cost)


def main():
    trainer = Trainer()
    # trainer.train(2000, progress=False)
    # trainer.test_defense()
    # trainer.total_test1()
    # trainer.test_defense_one()
    # trainer.BlackBoxAttackTest()
    trainer.test_time_cost()


if __name__ == "__main__":
    main()
