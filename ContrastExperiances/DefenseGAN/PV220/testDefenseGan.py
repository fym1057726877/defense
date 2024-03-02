import os
import torch
import logging
from ContrastExperiances.DefenseGAN.reconstruct import GradientDescentReconstruct
from ContrastExperiances.DefenseGAN.models import ConvGenerator
from utils import get_project_path, simple_test_defense
from data.VERA_PV220 import testloader


class DirectTrainActorReconstruct:
    def __init__(self):
        super(DirectTrainActorReconstruct, self).__init__()
        self.device = "cuda"
        self.model = ConvGenerator(latent_dim=128).to(self.device)
        self.save_path = os.path.join(
            get_project_path(project_name="Defense"),
            "ContrastExperiances",
            "DefenseGAN",
            "PV220",
            "ConvGenerator.pth"
        )
        self.model.load_state_dict((torch.load(self.save_path)))

        self.testloader = testloader

    def total_test(self):
        torch.manual_seed(10)
        logging.basicConfig(filename="./logs/defensegan_test_pv220.logs", filemode="w", level=logging.INFO, format="")
        classifier_name_list = [
            "FV_CNN_PV220",
            "PV_CNN_PV220",
            "LightweightFVCNN_PV220",
            "FVRAS_Net_PV220",
            "ModelB_PV220",
            # "Swintransformer_PV220",
            # "Res2Net_PV220",
            # "Vit_PV220"
        ]
        for classifier_name in classifier_name_list:
            log1 = simple_test_defense(defense_model=self.model, defense_fn=defense, attack_type="FGSM", eps=0.03,
                                       classifier_name=classifier_name, progress=False, device=self.device)
            print(log1)
            logging.info(log1)
            log2 = simple_test_defense(defense_model=self.model, defense_fn=defense, attack_type="FGSM", eps=0.05,
                                       classifier_name=classifier_name, progress=False, device=self.device)
            print(log2)
            logging.info(log2)
            log3 = simple_test_defense(defense_model=self.model, defense_fn=defense, attack_type="PGD", eps=0.03,
                                       classifier_name=classifier_name, progress=False, device=self.device)
            print(log3)
            logging.info(log3)
            log4 = simple_test_defense(defense_model=self.model, defense_fn=defense, attack_type="PGD", eps=0.05,
                                       classifier_name=classifier_name, progress=False, device=self.device)
            print(log4)
            logging.info(log4)


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
    worker.total_test()


if __name__ == "__main__":
    testDirectActor()
