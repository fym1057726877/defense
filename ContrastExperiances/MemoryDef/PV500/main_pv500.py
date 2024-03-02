import os
import torch
import logging
from data.FV500 import trainloader, testloader
from tqdm import tqdm
from time import time
from utils import get_project_path, simple_test_defense, print_time_consumed, accuracy, get_time_cost
from ContrastExperiances.MemoryDef.models import MemoryDefense
from classifier.models import Resnet50, load_ModelB_FV500


class Trainer:
    def __init__(self):
        super(Trainer, self).__init__()
        self.device = "cuda"

        self.model = MemoryDefense(
            in_channel=1,
            channel=256,
            n_res_block=2,
            n_res_channel=64,
            num_memories=1000,
            num_classes=500,
            feature_channel=4,
            threshold=None,
            sparse=True
        ).to(self.device)
        self.save_path = os.path.join(
            get_project_path(project_name="Defense"), "ContrastExperiances", "MemoryDef", "PV500", "memorydef_pv500.pth")
        self.model.load_state_dict(torch.load(self.save_path))

        self.trainloader = trainloader
        self.testloader = testloader

        self.classifier_name = "ModelB_FV500"
        self.classifierB = load_ModelB_FV500(device=self.device)

    def train(self, epochs, progress=False):
        start = time()

        logging.basicConfig(filename="./log/memorydef_pv500_train.logs", filemode="w", level=logging.INFO, format="")
        optimizer_ae = self.model.config_optim(lr=3e-4)
        best_acc = 0

        for epoch in range(epochs):
            count = len(self.trainloader)
            epoch_loss_ae = 0
            self.model.train()
            if progress:
                iter_object = tqdm(enumerate(self.trainloader), desc=f"train step {epoch + 1}/{epochs}", total=count)
            else:
                iter_object = enumerate(self.trainloader)
            for step, (img, lab) in iter_object:
                optimizer_ae.zero_grad()
                img, lab = img.to(self.device), lab.to(self.device)
                ae_loss = self.model.training_losses(img, lab)
                ae_loss.backward()
                optimizer_ae.step()
                epoch_loss_ae += ae_loss

            torch.save(self.model.state_dict(), self.save_path)
            with torch.no_grad():
                epoch_loss_ae = round(float(epoch_loss_ae / count), 8)
                nor_acc, rec_acc = self.test_rec_acc()
                print(f"epoch_loss:{epoch_loss_ae:.8f}   nor_acc:{nor_acc:.4f}   rec_acc:{rec_acc:.4f}")
                # log = simple_test_defense(defense_model=self.model, attack_type="FGSM", eps=0.03, defense_fn=defense,
                #                           classifier_name=self.classifier_name, progress=False,
                #                           print_info=True if epoch == 0 else False)
                # if log["RAvAcc"] > best_acc:
                #     print("saving the best model")
                #     torch.save(self.model.state_dict(), self.save_path)
                #     best_acc = log["RAvAcc"]
                # log["best_acc"] = best_acc
                # log = {"epoch": epoch + 1,
                #        "ae_loss": f"{epoch_loss_ae:.8f}",
                #        **log}
                # print(log)
                # logging.info(log)

        end = time()
        seconds = int(end - start)
        print_time_consumed(seconds)

    def test_rec_acc(self):
        total, nor, rec = 0, 0, 0
        for i, (x, y) in enumerate(self.testloader):
            x, y = x.to(self.device), y.to(self.device)
            rec_x = self.model(x, y)["rec_x"]
            pred_nor = self.classifierB(x)
            pred_rec = self.classifierB(rec_x)
            total += x.shape[0]
            nor += accuracy(pred_nor, y)
            rec += accuracy(pred_rec, y)
        nor = nor / total
        rec = rec / total
        return nor, rec

    def total_test1(self):
        classifier_name_list = [
            "Res2Net_FV500",
            "Vit_FV500",
            "SwinTransformer_FV500",
        ]
        for classifier_name in classifier_name_list:
            log1 = simple_test_defense(defense_model=self.model, attack_type="FGSM", eps=0.2, defense_fn=defense,
                                       testloader=self.testloader, progress=False,
                                       classifier_name=classifier_name, device=self.device)
            print(log1)
            log2 = simple_test_defense(defense_model=self.model, attack_type="PGD", eps=0.2, defense_fn=defense,
                                       testloader=self.testloader, progress=False,
                                       classifier_name=classifier_name, device=self.device)
            print(log2)

    def total_test(self):
        torch.manual_seed(10)
        logging.basicConfig(filename="./log/memdef_test_white.logs", filemode="w", level=logging.INFO, format="")
        classifier_name_list = [
            "FV_CNN_FV500",
            "PV_CNN_FV500",
            "LightweightFVCNN_FV500",
            "FVRAS_Net_FV500",
            "ModelB_FV500",
        ]
        for classifier_name in classifier_name_list:
            log1 = simple_test_defense(defense_model=self.model, attack_type="FGSM", eps=0.3, defense_fn=defense,
                                       testloader=self.testloader, progress=False,
                                       classifier_name=classifier_name, device=self.device)
            print(log1)
            logging.info(log1)
            log2 = simple_test_defense(defense_model=self.model, attack_type="PGD", eps=0.3, defense_fn=defense,
                                       testloader=self.testloader, progress=False,
                                       classifier_name=classifier_name, device=self.device)
            print(log2)
            logging.info(log2)

    def BlackBoxAttackTest(self):
        torch.manual_seed(10)
        logging.basicConfig(filename="./log/memdef_test_black.logs", filemode="w", level=logging.INFO, format="")
        classifier_name_list = [
            "FV_CNN_FV500",
            "PV_CNN_FV500",
            "LightweightFVCNN_FV500",
            "FVRAS_Net_FV500",
            "ModelB_FV500",
            # "Vit_FV500",
            "Res2Net_FV500",
            "SwinTransformer_FV500",
        ]
        for classifier_name in classifier_name_list:
            log1 = simple_test_defense(defense_model=self.model, attack_type="SPSA", eps=0.3, defense_fn=defense,
                                       testloader=self.testloader, progress=False,
                                       classifier_name=classifier_name, device=self.device)
            print(log1)
            logging.info(log1)

    def test_time_cost(self):
        classifier_name = "PV_CNN_FV500"
        time_cost = get_time_cost(defense_model=self.model, classifier_name=classifier_name, defense_fn=defense,
                                  attack_type="FGSM", eps=0.3, progress=False, device=self.device)
        print(time_cost)


def defense(x, memdef):
    classfierA = Resnet50(num_classes=500).to(x.device)
    save_path = os.path.join(get_project_path("Defense"), "ContrastExperiances", "MemoryDef", "PV500", f"Resnet50_PV500.pth")
    classfierA.load_state_dict(torch.load(save_path))
    classfierA.eval()
    label = classfierA(x).max(dim=1)[1]
    rec = memdef(x, label)["rec_x"]
    return rec


def main():
    trainer = Trainer()
    # trainer.train(2000)
    # trainer.test()
    # trainer.test_defense()
    # print(trainer.test_rec_acc())
    # trainer.total_test()
    # trainer.total_test1()
    # trainer.BlackBoxAttackTest()
    trainer.test_time_cost()

if __name__ == "__main__":
    main()

