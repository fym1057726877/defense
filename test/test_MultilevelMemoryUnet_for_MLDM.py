
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from ours.diffusion.models import MultilevelMemoryUnet
from LDM.aemodel.autoencoder import MemoryAutoencoder
from ours.diffusion.respace import iter_denoise
from classifier.attackClassifier import generateAdvImage
from classifier.models import getDefinedClsModel
from data.TJ_PV600 import testloader
from utils import *


class DirectTrainActorReconstruct:
    def __init__(
            self,
            target_classifier_name="ModelB",
            attack_type="FGSM",   # "RandFGSM", "FGSM", "PGD", "HSJA"
            eps=0.03,
            adv_imgs_folder_name="adv_imgs_ModelB_PV600"
    ):
        super(DirectTrainActorReconstruct, self).__init__()
        self.device = "cuda"

        self.attack_type = attack_type
        self.eps = eps
        self.batch_size = 50
        self.num_classes = 600

        self.unet = MultilevelMemoryUnet(
            in_channels=4,
            model_channels=32,
            out_channels=4,
            channel_mult=(1, 2, 2),
            num_res_blocks=2,
            features=1024,
            num_memories=600,
            threshold=1e-5
        ).to(self.device)
        self.save_path = os.path.join(
            get_project_path(project_name="Defense"), "pretrained", "Multi_Memory_Unet_for_LDM.pth")
        self.unet.load_state_dict(torch.load(self.save_path))

        self.autoencoder = MemoryAutoencoder(
            in_ch=1,
            ch=128,
            z_ch=4,
            out_ch=1,
            embed_dim=4,
            ch_mult=(1, 2, 4),
            resolution=64,
            num_res_blocks=2,
            attn_resolutions=(4, 2, 1),
            dropout=0.0,
            double_z=False,
            num_memories=600,
            threshold=1e-4,
            recloss_type="l1"
        ).to(self.device)
        self.ae_path = os.path.join(get_project_path(project_name="Defense"), "pretrained", "MemoryAE_LDM.pth")
        self.autoencoder.load_state_dict(torch.load(self.ae_path))

        # classifier B
        # dataset_name = "Handvein"
        # dataset_name = "Handvein3"
        dataset_name = "Fingervein2"
        self.target_classifier_name = target_classifier_name
        self.target_classifier = getDefinedClsModel(
            dataset_name=dataset_name,
            model_name=self.target_classifier_name,
            num_classes=self.num_classes,
            device=self.device
        )
        target_classifier_path = os.path.join(
            get_project_path(project_name="Defense"), "pretrained", f"{self.target_classifier_name}.pth")

        self.target_classifier.load_state_dict(torch.load(target_classifier_path))

        # adversial
        self.adv_path = os.path.join(
            get_project_path(project_name="Defense"),
            "data",
            adv_imgs_folder_name,
            f"{self.num_classes}_{self.target_classifier_name}_{self.attack_type}_{self.eps}.pth"
        )

        self.attack_dataloader = testloader

    def defense(self, img):
        h = self.autoencoder.encode(img)
        final_h = iter_denoise(self.unet, imgs=h, t=100)
        final_z = self.autoencoder.encode_encode(final_h)[0]
        rec = self.autoencoder.decode(final_z)
        return rec

    def test(self, progress=False):
        advDataloader = self.getAdvDataLoader(
            progress=True,
            shuffle=True
        )

        self.target_classifier.eval()
        self.unet.eval()

        def accuracy(y, y1):
            return (y.max(dim=1)[1] == y1).sum().data.item()

        normal_acc, rec_acc, adv_acc, rec_adv_acc, num = 0, 0, 0, 0, 0
        total_num = len(advDataloader)

        iterObject = enumerate(advDataloader)
        if progress:
            iterObject = tqdm(iterObject, total=total_num)

        for i, (img, adv_img, label) in iterObject:
            img, label = img.to(self.device), label.to(self.device)
            num += label.size(0)

            normal_y = self.target_classifier(img)
            normal_acc += accuracy(normal_y, label)
            # print(f"normal_acc:{normal_acc}/{num}")

            rec_img = self.defense(img)
            rec_y = self.target_classifier(rec_img)
            rec_acc += accuracy(rec_y, label)
            # print(f"rec_acc:{rec_acc}/{num}")

            adv_img = adv_img.to(self.device)
            adv_y = self.target_classifier(adv_img)
            adv_acc += accuracy(adv_y, label)
            # print(f"adv_acc:{adv_acc}/{num}")

            rec_adv_img = self.defense(adv_img)
            rec_adv_y = self.target_classifier(rec_adv_img)
            rec_adv_acc += accuracy(rec_adv_y, label)
            # print(f"rec_adv_acc:{rec_adv_acc}/{num}")

        print(f"-------------------------------------------------\n"
              f"test result with attacktype = {self.attack_type} and eps={self.eps}:\n"
              f"NorAcc:{torch.true_divide(normal_acc, num).item():.3f}\n"
              f"RecAcc:{torch.true_divide(rec_acc, num).item():.3f}\n"
              f"AdvAcc:{torch.true_divide(adv_acc, num).item():.3f}\n"
              f"RAvAcc:{torch.true_divide(rec_adv_acc, num).item():.3f}\n"
              f"-------------------------------------------------")

        return [normal_acc/num, rec_acc/num, adv_acc/num, rec_adv_acc/num]

    def getAdvDataLoader(
            self,
            progress=False,
            shuffle=False
    ):
        if os.path.exists(self.adv_path):
            data_dict = torch.load(self.adv_path)
        else:
            data_dict = generateAdvImage(
                classifier=self.target_classifier,
                attack_dataloader=self.attack_dataloader,
                attack_type=self.attack_type,
                eps=self.eps,
                progress=progress,
                savepath=self.adv_path
            )
        normal_data = data_dict["normal"]
        adv_data = data_dict["adv"]
        label = data_dict["label"]
        dataloder = DataLoader(TensorDataset(normal_data, adv_data, label), batch_size=self.batch_size, shuffle=shuffle)
        return dataloder


def total_test(target_classifier_name, defense_model_name):
    torch.manual_seed(10)
    attack_types = ['FGSM', 'PGD']
    epsilons = [0.03, 0.1, 0.3]
    acc_info = dict()
    for at in attack_types:
        for e in epsilons:
            directActorRec = DirectTrainActorReconstruct(
                target_classifier_name=target_classifier_name,
                attack_type=at,
                eps=e,
            )
            acc_list = directActorRec.test(progress=False)
            acc_info[f"{at}_{e}"] = acc_list
    savepath = os.path.join(
        get_project_path("Defense"), "defense_result", f"{target_classifier_name}_{defense_model_name}.pth")
    torch.save(acc_info, savepath)


def simple_test():
    # seed = 1  # the seed for random function
    torch.manual_seed(10)
    attack_type = 'FGSM'
    # attack_type = 'PGD'
    # attack_type = 'RandFGSM'
    # attack_type = 'HSJA'
    if attack_type == 'FGSM':
        # eps = 0.03
        eps = 0.1
        # eps = 0.3
    elif attack_type == 'RandFGSM':
        # eps = 0.03
        # eps = 0.1
        eps = 0.3
    elif attack_type == "PGD":
        # eps = 0.03
        # eps = 0.1
        eps = 0.3
    else:
        raise NotImplementedError("Unknown attack")
    # classifier_name = "Resnet18"
    # classifier_name = "GoogleNet"
    classifier_name = "ModelB"
    # classifier_name = "MSMDGANetCnn_wo_MaxPool"
    # classifier_name = "Tifs2019Cnn_wo_MaxPool"
    # classifier_name = "FVRASNet_wo_Maxpooling"
    # classifier_name = "LightweightDeepConvNN"

    directActorRec = DirectTrainActorReconstruct(
        target_classifier_name=classifier_name,
        attack_type=attack_type,
        eps=eps,
    )
    directActorRec.test(progress=True)


if __name__ == "__main__":
    # total_test(target_classifier_name="ModelB", defense_model_name="MultilevelMemoryUnet_t=30")
    simple_test()
