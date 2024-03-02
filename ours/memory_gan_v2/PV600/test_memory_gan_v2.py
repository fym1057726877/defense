import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from ours.memory_gan_v2.PV600.models import MemoryGAN_v2
from classifier.attackClassifier import generateAdvImage
from classifier.models import getDefinedClsModel
from data.TJ_PV600 import testloader
from utils import *


def accuracy(y, y1):
    return (y.max(dim=1)[1] == y1).sum().data.item()


class DirectTrainActorReconstruct:
    def __init__(
            self,
            target_classifier_name="ModelB",
            attack_type="FGSM",  # "RandFGSM", "FGSM", "PGD", "HSJA"
            eps=0.03,
            adv_imgs_folder_name="adv_imgs_ModelB_PV600"
    ):
        super(DirectTrainActorReconstruct, self).__init__()
        self.device = "cuda"

        self.attack_type = attack_type
        self.eps = eps
        self.batch_size = 50
        self.num_classes = 600

        self.autoencoder = MemoryGAN_v2(
            in_channel=1,
            channel=256,
            n_res_block=2,
            n_res_channel=64,
            num_memories=600,
            features_channels=4,
            memory_threshold=None,
            sparse=True,
            resolution=64
        ).to(self.device)
        self.save_path = os.path.join(
            get_project_path(project_name="Defense"), "ours", "memory_gan_v2", "memory_gan_v2.pth")

        # self.save_path = os.path.join(
        #     get_project_path(project_name="Defense"), "ours", "memory_gan_v2", "best_model.pth")
        self.autoencoder.load_state_dict(torch.load(self.save_path))

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
        rec = self.autoencoder(img)[0]
        return rec

    def test(self, progress=False):
        advDataloader = self.getAdvDataLoader(
            progress=True,
            shuffle=True
        )

        self.target_classifier.eval()
        self.autoencoder.eval()

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

            # draw_img_groups([img, rec_img, adv_img, rec_adv_img])
            # return

        print(f"-------------------------------------------------\n"
              f"test result:\n"
              f"test result with attacktype = {self.attack_type} and eps={self.eps}:\n"
              f"NorAcc:{torch.true_divide(normal_acc, num).item():.3f}\n"
              f"RecAcc:{torch.true_divide(rec_acc, num).item():.3f}\n"
              f"AdvAcc:{torch.true_divide(adv_acc, num).item():.3f}\n"
              f"RAvAcc:{torch.true_divide(rec_adv_acc, num).item():.3f}\n"
              f"-------------------------------------------------")

        return [normal_acc / num, rec_acc / num, adv_acc / num, rec_adv_acc / num]

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
    # attack_type = 'CW'
    if attack_type == 'FGSM':
        # eps = 0.01
        # eps = 0.03
        eps = 0.1
        # eps = 0.15
        # eps = 0.2
        # eps = 0.3
    elif attack_type == 'RandFGSM':
        # eps = 0.03
        # eps = 0.1
        eps = 0.3
    elif attack_type == "PGD":
        eps = 0.03
        # eps = 0.1
        # eps = 0.3
    elif attack_type == "CW":
        eps = 100
        # eps = 0.1
        # eps = 0.3
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
    return directActorRec.test(progress=True)


def simple_test_defense(
        *,
        defense_model,
        classifier=None,
        classifier_name="ModelB",
        num_classes=600,
        classifier_path=None,
        eps=0.1,
        attack_type="FGSM",
        adv_imgs_path=None,
        device="cuda",
        progress=True
):
    assert attack_type in ["FGSM", "RandFGSM", "PGD"]
    if classifier is None:
        classifier = getDefinedClsModel(
            model_name=classifier_name,
            num_classes=num_classes,
            device=device
        )
        if classifier_path is None:
            classifier_path = os.path.join(
                get_project_path(project_name="Defense"), "pretrained", f"{classifier_name}.pth")
        print(f"load classifier {classifier_name} state_dict from {classifier_path}")
        classifier.load_state_dict(torch.load(classifier_path))

    # adversial
    if adv_imgs_path is None:
        adv_imgs_path = os.path.join(get_project_path(project_name="Defense"), "data", "adv_imgs_ModelB_PV600",
                                     f"{num_classes}_{classifier_name}_{attack_type}_{eps}.pth")
    print(f"test adv_imgs_path:{adv_imgs_path}")

    def getAdvDataLoader(shuffle=False, batch_size=50):
        if os.path.exists(adv_imgs_path):
            data_dict = torch.load(adv_imgs_path)
        else:
            data_dict = generateAdvImage(
                classifier=classifier,
                attack_dataloader=testloader,
                attack_type=attack_type,
                eps=eps,
                progress=False,
                savepath=adv_imgs_path
            )
        normal_data = data_dict["normal"]
        adv_data = data_dict["adv"]
        label = data_dict["label"]
        dataloder = DataLoader(TensorDataset(normal_data, adv_data, label), batch_size=batch_size, shuffle=shuffle)
        return dataloder

    def defense(x):
        return defense_model(x)[0]

    advDataloader = getAdvDataLoader(shuffle=True)

    classifier.eval()
    defense_model.eval()

    normal_acc, rec_acc, adv_acc, rec_adv_acc, num = 0, 0, 0, 0, 0
    total_num = len(advDataloader)

    iterObject = enumerate(advDataloader)
    if progress:
        iterObject = tqdm(iterObject, total=total_num, desc="testing")

    for i, (img, adv_img, label) in iterObject:
        img, label = img.to(device), label.to(device)
        num += label.size(0)

        normal_y = classifier(img)
        normal_acc += accuracy(normal_y, label)
        # print(f"normal_acc:{normal_acc}/{num}")

        rec_img = defense(img)
        rec_y = classifier(rec_img)
        rec_acc += accuracy(rec_y, label)
        # print(f"rec_acc:{rec_acc}/{num}")

        adv_img = adv_img.to(device)
        adv_y = classifier(adv_img)
        adv_acc += accuracy(adv_y, label)
        # print(f"adv_acc:{adv_acc}/{num}")

        rec_adv_img = defense(adv_img)
        rec_adv_y = classifier(rec_adv_img)
        rec_adv_acc += accuracy(rec_adv_y, label)
        # print(f"rec_adv_acc:{rec_adv_acc}/{num}")

    NorAcc = round(normal_acc / num, 3)
    RecAcc = round(rec_acc / num, 3)
    AdvAcc = round(adv_acc / num, 3)
    RAvAcc = round(rec_adv_acc / num, 3)
    log = {"attacktype": attack_type,
           "eps": eps,
           "NorAcc": NorAcc,
           "RecAcc": RecAcc,
           "AdvAcc": AdvAcc,
           "RAvAcc": RAvAcc}

    return [NorAcc, RecAcc, AdvAcc, RAvAcc], log


if __name__ == "__main__":
    # total_test(target_classifier_name="ModelB", defense_model_name="MemoryAE_in_Paper")
    # simple_test()
    model = MemoryGAN_v2(
        in_channel=1,
        channel=256,
        n_res_block=2,
        n_res_channel=64,
        num_memories=600,
        features_channels=4,
        memory_threshold=None,
        sparse=True,
        resolution=64
    ).to("cuda")
    # save_path = os.path.join(get_project_path(project_name="Defense"), "ours", "memory_gan_v2", "memory_gan_v2.pth")
    save_path = os.path.join(get_project_path(project_name="Defense"), "ours", "memory_gan_v2", "best_model.pth")
    model.load_state_dict(torch.load(save_path))
    accs, log = simple_test_defense(defense_model=model, attack_type="FGSM", eps=0.15)
    print(accs)
    print(log)
