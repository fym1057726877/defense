import os
import torch
import logging
from ContrastExperiances.TransGan.TransGanModel import SwinTransGenerator
from ContrastExperiances.TransGan.Actor import Resnet34Actor
from utils import get_project_path, simple_test_defense, get_time_cost
from data.TJ_PV600 import testloader


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


class TransGanDefense(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cuda"
        self.encoder = Resnet34Actor().to(self.device)
        self.encoder_path = os.path.join(
            get_project_path(project_name="Defense"),
            "ContrastExperiances",
            "TransGan",
            "PV600",
            # "checkpoint",
            "resnet34actor.pth"
        )
        self.encoder.load_state_dict(torch.load(self.encoder_path))

        self.generater = SwinTransGenerator(
            embed_dim=g_embed_dim,
            bottom_width=bottom_width,
            bottom_height=bottom_height,
            window_size=window_size,
            depth=g_depths,
            is_local=is_local,
            is_peg=is_peg
        ).to(self.device)
        self.generater_path = os.path.join(
            get_project_path(project_name="Defense"),
            "ContrastExperiances",
            "TransGan",
            "PV600",
            "checkpoint",
            "TransGenerator.pth"
        )
        self.generater.load_state_dict((torch.load(self.generater_path)))

    def forward(self, img):
        z = self.encoder(img)
        rec = self.generater(z)
        return rec


class Worker:
    def __init__(self,):
        super(Worker, self).__init__()
        self.device = "cuda"
        self.model = TransGanDefense().to(self.device)
        self.testloader = testloader

    def test_defense_one(self):
        classifier_name = "SwinTransformer_PV600"
        log = simple_test_defense(defense_model=self.model, attack_type="FGSM", eps=0.015,
                                  testloader=self.testloader,
                                  classifier_name=classifier_name, device=self.device)
        print(log)

    def test_defense(self):
        # classifier_name = "ModelB_PV600"
        # classifier_name = "Vit_PV600"
        # classifier_name = "Res2Net_PV600"
        classifier_name = "SwinTransformer_PV600"

        # classifier_name = "FV_CNN_PV600"
        # classifier_name = "PV_CNN_PV600"
        # classifier_name = "LightweightFVCNN_PV600"
        # classifier_name = "FVRAS_Net_PV600"

        log1 = simple_test_defense(defense_model=self.model, attack_type="FGSM", eps=0.01,
                                   testloader=self.testloader,
                                   classifier_name=classifier_name, device=self.device)
        print(log1)
        log2 = simple_test_defense(defense_model=self.model, attack_type="FGSM", eps=0.03,
                                   testloader=self.testloader,
                                   classifier_name=classifier_name, device=self.device)
        print(log2)
        log3 = simple_test_defense(defense_model=self.model, attack_type="PGD", eps=0.01,
                                   testloader=self.testloader,
                                   classifier_name=classifier_name, device=self.device)
        print(log3)
        log4 = simple_test_defense(defense_model=self.model, attack_type="PGD", eps=0.015,
                                   testloader=self.testloader,
                                   classifier_name=classifier_name, device=self.device)
        print(log4)

    def total_test(self):
        classifier_name_list = [
            "FV_CNN_PV600",
            "PV_CNN_PV600",
            "LightweightFVCNN_PV600",
            "FVRAS_Net_PV600"
        ]
        logging.basicConfig(filename="./logs/TransGan_test.logs", filemode="w", level=logging.INFO, format="")
        for classifier_name in classifier_name_list:
            log1 = simple_test_defense(defense_model=self.model, attack_type="FGSM", eps=0.03,
                                       testloader=self.testloader,
                                       classifier_name=classifier_name, device=self.device)
            print(log1)
            logging.info(log1)
            log2 = simple_test_defense(defense_model=self.model, attack_type="FGSM", eps=0.05,
                                       testloader=self.testloader,
                                       classifier_name=classifier_name, device=self.device)
            print(log2)
            logging.info(log2)
            log3 = simple_test_defense(defense_model=self.model, attack_type="PGD", eps=0.03,
                                       testloader=self.testloader,
                                       classifier_name=classifier_name, device=self.device)
            print(log3)
            logging.info(log3)
            log4 = simple_test_defense(defense_model=self.model, attack_type="PGD", eps=0.05,
                                       testloader=self.testloader,
                                       classifier_name=classifier_name, device=self.device)
            print(log4)
            logging.info(log4)

    def test_time_cost(self):
        classifier_name = "PV_CNN_FV500"
        time_cost = get_time_cost(defense_model=self.model, classifier_name=classifier_name,
                                  attack_type="FGSM", eps=0.1, progress=False, device=self.device)
        print(time_cost)


def main():
    worker = Worker()
    # worker.total_test()
    # worker.test_defense()
    # worker.test_defense_one()
    worker.test_time_cost()

if __name__ == "__main__":
    main()
