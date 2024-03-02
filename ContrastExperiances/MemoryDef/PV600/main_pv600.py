import os
import torch
import logging
from data.TJ_PV600 import trainloader, testloader
from tqdm import tqdm
from time import time
from utils import get_project_path, simple_test_defense, print_time_consumed, accuracy, get_time_cost
from ContrastExperiances.MemoryDef.models import MemoryDefense
from classifier.models import Resnet50, load_ModelB_PV600


class Trainer:
    def __init__(self):
        super(Trainer, self).__init__()
        self.device = "cuda"

        self.model = MemoryDefense(
            in_channel=1,
            channel=256,
            n_res_block=2,
            n_res_channel=64,
            num_memories=1200,
            num_classes=600,
            feature_channel=4,
            threshold=None,
            sparse=True
        ).to(self.device)
        self.save_path = os.path.join(
            get_project_path(project_name="Defense"), "ContrastExperiances", "MemoryDef", "PV600", "memorydef_pv600.pth")
        self.model.load_state_dict(torch.load(self.save_path))

        self.trainloader = trainloader
        self.testloader = testloader

        self.classifier_name = "ModelB_PV600"
        self.classifierB = load_ModelB_PV600(device=self.device)

    def train(self, epochs, progress=False):
        start = time()

        logging.basicConfig(filename="./log/memorydef_pv600_train.logs", filemode="w", level=logging.INFO, format="")
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
                if rec_acc > 0.90 or epoch > 999:
                    return
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
        torch.manual_seed(10)
        classifier_name_list = [
            "Res2Net_PV600",
            "Vit_PV600",
            "SwinTransformer_PV600",
        ]
        for classifier_name in classifier_name_list:
            log1 = simple_test_defense(defense_model=self.model, attack_type="FGSM", eps=0.01, defense_fn=defense,
                                       testloader=self.testloader, progress=False,
                                       classifier_name=classifier_name, device=self.device)
            print(log1)
            log2 = simple_test_defense(defense_model=self.model, attack_type="FGSM", eps=0.015, defense_fn=defense,
                                       testloader=self.testloader, progress=False,
                                       classifier_name=classifier_name, device=self.device)
            print(log2)
            log3 = simple_test_defense(defense_model=self.model, attack_type="PGD", eps=0.01, defense_fn=defense,
                                       testloader=self.testloader, progress=False,
                                       classifier_name=classifier_name, device=self.device)
            print(log3)
            log4 = simple_test_defense(defense_model=self.model, attack_type="PGD", eps=0.015, defense_fn=defense,
                                       testloader=self.testloader, progress=False,
                                       classifier_name=classifier_name, device=self.device)
            print(log4)

    def total_test(self):
        torch.manual_seed(10)
        logging.basicConfig(filename="./log/memorydef_white_test.logs", filemode="w", level=logging.INFO, format="")
        classifier_name_list = [
            "FV_CNN_PV600",
            "PV_CNN_PV600",
            "LightweightFVCNN_PV600",
            "FVRAS_Net_PV600",
            "ModelB_PV600",
        ]
        for classifier_name in classifier_name_list:
            log1 = simple_test_defense(defense_model=self.model, attack_type="FGSM", eps=0.03, defense_fn=defense,
                                       testloader=self.testloader, progress=False,
                                       classifier_name=classifier_name, device=self.device)
            print(log1)
            logging.info(log1)
            log2 = simple_test_defense(defense_model=self.model, attack_type="FGSM", eps=0.05, defense_fn=defense,
                                       testloader=self.testloader, progress=False,
                                       classifier_name=classifier_name, device=self.device)
            print(log2)
            logging.info(log2)
            log3 = simple_test_defense(defense_model=self.model, attack_type="PGD", eps=0.03, defense_fn=defense,
                                       testloader=self.testloader, progress=False,
                                       classifier_name=classifier_name, device=self.device)
            print(log3)
            logging.info(log3)
            log4 = simple_test_defense(defense_model=self.model, attack_type="PGD", eps=0.05, defense_fn=defense,
                                       testloader=self.testloader, progress=False,
                                       classifier_name=classifier_name, device=self.device)
            print(log4)
            logging.info(log4)

    def BlackBoxAttackTest(self):
        torch.manual_seed(10)
        logging.basicConfig(filename="logs/memorydef_black_test.logs", filemode="w", level=logging.INFO, format="")
        classifier_name_list = [
            "FV_CNN_PV600",
            # "PV_CNN_PV600",
            "LightweightFVCNN_PV600",
            # "FVRAS_Net_PV600",
            "ModelB_PV600",
            "Vit_PV600",
            "Res2Net_PV600",
            "SwinTransformer_PV600",
        ]
        for classifier_name in classifier_name_list:
            log1 = simple_test_defense(defense_model=self.model, attack_type="SPSA", eps=0.1, defense_fn=defense,
                                       testloader=self.testloader, progress=True,
                                       classifier_name=classifier_name, device=self.device)
            print(log1)
            logging.info(log1)

        log2 = simple_test_defense(defense_model=self.model, attack_type="SPSA", eps=0.1, defense_fn=defense,
                                   testloader=self.testloader, progress=True,
                                   classifier_name="PV_CNN_PV600", device=self.device)
        print(log2)
        logging.info(log2)

        log3 = simple_test_defense(defense_model=self.model, attack_type="SPSA", eps=0.1, defense_fn=defense,
                                   testloader=self.testloader, progress=True,
                                   classifier_name="FVRAS_Net_PV600", device=self.device)
        print(log3)
        logging.info(log3)

    def test_time_cost(self):
        classifier_name = "PV_CNN_PV600"
        time_cost = get_time_cost(defense_model=self.model, classifier_name=classifier_name, defense_fn=defense,
                                  attack_type="FGSM", eps=0.03, progress=False, device=self.device)
        print(time_cost)


def defense_memdef(x, memdef):
    classfierA = Resnet50(num_classes=600).to(x.device)
    save_path = os.path.join(get_project_path("Defense"), "ContrastExperiances", "MemoryDef", "PV600", f"Resnet50_PV600.pth")
    classfierA.load_state_dict(torch.load(save_path))
    classfierA.eval()
    label = classfierA(x).max(dim=1)[1]
    rec = memdef(x, label)["rec_x"]
    return rec


def main():
    trainer = Trainer()
    # trainer.train(1500)
    # trainer.test()
    # trainer.test_defense()
    # trainer.total_test()
    # trainer.total_test1()
    # trainer.BlackBoxAttackTest()
    # print(trainer.test_rec_acc())
    trainer.test_time_cost()

if __name__ == "__main__":
    main()

