import os
import torch
import logging
from tqdm import tqdm
from time import time
from utils import get_project_path, print_time_consumed, draw_img_groups, simple_test_defense, get_time_cost
from data.FV500 import trainloader, testloader
from ours.models import MlpMemoryGAN, MemoryGAN_v2


class Trainer:
    def __init__(
            self,
    ):
        super().__init__()
        self.device = "cuda"
        self.model = MlpMemoryGAN(
            in_channel=1,
            channel=256,
            n_res_block=2,
            n_res_channel=64,
            num_memories=1000,
            features_channels=4,
            memory_threshold=None,
            sparse=True,
            resolution=64,
            other_dims=1
        ).to(self.device)
        # self.model = MemoryGAN_v2(
        #     in_channel=1,
        #     channel=256,
        #     n_res_block=2,
        #     n_res_channel=64,
        #     num_memories=300,
        #     features_channels=4,
        #     memory_threshold=None,
        #     sparse=True,
        #     resolution=64,
        # ).to(self.device)
        self.save_path = os.path.join(get_project_path("Defense"), "ours", "attn_memory_gan", "FV500", "attn_memory_gan_FV500_1.pth")
        self.model.load_state_dict(torch.load(self.save_path))

        self.trainloader = trainloader
        self.testloader = testloader

        # self.classifier_name = "SwinTransformer_FV500"
        self.classifier_name = "LightweightFVCNN_FV500"

    def train(self, epochs, progress=False):
        start = time()

        logging.basicConfig(filename="./logs/attn_memory_gan_train_FV500.logs", filemode="w", level=logging.INFO, format="")
        optimizer_ae, optimizer_disc = self.model.config_optimizer(lr=3e-4)
        best_acc = 0

        for epoch in range(epochs):
            count = len(self.trainloader)
            epoch_loss_ae, epoch_loss_disc = 0, 0
            self.model.train()
            if progress:
                iter_object = tqdm(enumerate(self.trainloader), desc=f"train step {epoch + 1}/{epochs}", total=count)
            else:
                iter_object = enumerate(self.trainloader)
            for step, (img, _) in iter_object:
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

            # torch.save(self.model.state_dict(), self.save_path)
            with torch.no_grad():
                epoch_loss_ae = round(float(epoch_loss_ae / count), 8)
                epoch_loss_disc = round(float(epoch_loss_disc / count), 8)
                log = simple_test_defense(defense_model=self.model, attack_type="PGD", eps=0.03, testloader=self.testloader,
                                          classifier_name=self.classifier_name, progress=False,
                                          print_info=True if epoch == 0 else False)
                if log["RAvAcc"] > best_acc:
                    print("saving the best model")
                    torch.save(self.model.state_dict(), self.save_path)
                    best_acc = log["RAvAcc"]
                log["best_acc"] = best_acc
                log = {"epoch": epoch + 1,
                       "ae_loss": f"{epoch_loss_ae:.8f}",
                       "disc_loss": f"{epoch_loss_disc}",
                       **log}
                print(log)
                logging.info(log)

        end = time()
        seconds = int(end - start)
        print_time_consumed(seconds)

    def test_rec(self):
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
                                  testloader=self.testloader, iters=64,
                                  classifier_name=classifier_name, device=self.device)
        print(log)

    def test_defense(self):
        torch.manual_seed(10)
        # classifier_name = "ModelB_FV500"
        # classifier_name = "Vit_FV500"
        # classifier_name = "Res2Net_FV500"
        # classifier_name = "SwinTransformer_FV500"

        # classifier_name = "FV_CNN_FV500"
        # classifier_name = "PV_CNN_FV500"
        classifier_name = "LightweightFVCNN_FV500"
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
        logging.basicConfig(filename="./logs/memory_gan_v2_test1.logs", filemode="w", level=logging.INFO, format="")
        classifier_name_list = [
            # "FV_CNN_FV500",
            # "PV_CNN_FV500",
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
            log1 = simple_test_defense(defense_model=self.model, attack_type="SPSA", eps=0.8,
                                       testloader=self.testloader, progress=True, iters=128,
                                       classifier_name=classifier_name, device=self.device, logits=True)
            print(log1)
            logging.info(log1)

    def test_time_cost(self):
        classifier_name = "PV_CNN_FV500"
        time_cost = get_time_cost(defense_model=self.model, classifier_name=classifier_name,
                                  attack_type="FGSM", eps=0.1, progress=False, device=self.device)
        print(time_cost)


def main():
    trainer = Trainer()
    # trainer.train(2000)
    # trainer.test_defense()
    # trainer.total_test()
    # trainer.total_test1()
    # trainer.test_defense_one()
    # trainer.BlackBoxAttackTest()
    trainer.test_time_cost()


if __name__ == "__main__":
    main()
