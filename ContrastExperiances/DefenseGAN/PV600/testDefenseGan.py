import os
import torch
import logging
from ContrastExperiances.DefenseGAN.reconstruct import GradientDescentReconstruct
from ContrastExperiances.DefenseGAN.models import ConvGenerator
from utils import get_project_path, simple_test_defense
from data.TJ_PV600 import testloader


class DirectTrainActorReconstruct:
    def __init__(self):
        super(DirectTrainActorReconstruct, self).__init__()
        self.device = "cuda"
        self.model = ConvGenerator(latent_dim=128).to(self.device)
        self.save_path = os.path.join(
            get_project_path(project_name="Defense"),
            "ContrastExperiances",
            "DefenseGAN",
            "PV600",
            "ConvGenerator.pth"
        )
        self.model.load_state_dict((torch.load(self.save_path)))

        self.testloader = testloader

    def total_test(self):
        classifier_name_list = [
            # "FV_CNN_PV600",
            # "PV_CNN_PV600",
            # "LightweightFVCNN_PV600",
            # "FVRAS_Net_PV600"，
            # "SwinTransformer_PV600",
            "Res2Net_PV600",
            "Vit_PV600"
        ]
        logging.basicConfig(filename="logs/DefenseGan_test.log", filemode="w", level=logging.INFO, format="")
        for classifier_name in classifier_name_list:
            # log1 = simple_test_defense(defense_model=self.model, attack_type="FGSM", eps=0.01,
            #                            testloader=self.testloader, defense_fn=defense,
            #                            classifier_name=classifier_name, device=self.device)
            # print(log1)
            # logging.info(log1)
            log2 = simple_test_defense(defense_model=self.model, attack_type="FGSM", eps=0.015,
                                       testloader=self.testloader, defense_fn=defense,
                                       classifier_name=classifier_name, device=self.device)
            print(log2)
            logging.info(log2)
            # log3 = simple_test_defense(defense_model=self.model, attack_type="PGD", eps=0.01,
            #                            testloader=self.testloader, defense_fn=defense,
            #                            classifier_name=classifier_name, device=self.device)
            # print(log3)
            # logging.info(log3)
            # log4 = simple_test_defense(defense_model=self.model, attack_type="PGD", eps=0.015,
            #                            testloader=self.testloader, defense_fn=defense,
            #                            classifier_name=classifier_name, device=self.device)
            # print(log4)
            # logging.info(log4)

    def BlackBoxAttactTest(self):
        torch.manual_seed(10)
        logging.basicConfig(filename="./logs/blackbox.logs", filemode="w", level=logging.INFO, format="")
        classifier_name_list = [
            "FV_CNN_PV600",
            "PV_CNN_PV600",
            "LightweightFVCNN_PV600",
            "FVRAS_Net_PV600",
            "ModelB_PV600",
            "Vit_PV600",
            "Res2Net_PV600",
            "SwinTransformer_PV600",
        ]
        for classifier_name in classifier_name_list:
            log1 = simple_test_defense(defense_model=self.model, attack_type="SPSA", eps=0.1,
                                       testloader=self.testloader, progress=True, defense_fn=defense,
                                       classifier_name=classifier_name, device=self.device)
            print(log1)
            logging.info(log1)

    def test_defense_one(self):
        torch.manual_seed(10)
        # classifier_name = "ModelB_PV600"
        # classifier_name = "Vit_PV600"
        # classifier_name = "Res2Net_PV600"
        # classifier_name = "SwinTransformer_PV600"

        # classifier_name = "FV_CNN_PV600"
        # classifier_name = "PV_CNN_PV600"
        # classifier_name = "LightweightFVCNN_PV600"
        classifier_name = "FVRAS_Net_PV600"
        log = simple_test_defense(defense_model=self.model, attack_type="SPSA", eps=0.2,
                                  testloader=self.testloader, progress=True, defense_fn=defense,
                                  classifier_name=classifier_name, device=self.device)
        print(log)


def defense(img, model):
    rec = GradientDescentReconstruct(
        img=img,
        device="cuda",
        generator=model,
        lr=0.03,
        L=2000,
        R=10,
        latent_dim=128,
    )
    return rec


def testDirectActor():
    worker = DirectTrainActorReconstruct()
    # worker.total_test()
    # worker.BlackBoxAttactTest()
    worker.test_defense_one()
if __name__ == "__main__":
    testDirectActor()
