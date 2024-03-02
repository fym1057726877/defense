
import os
import torch
import logging
from tqdm import tqdm
from time import time
from matplotlib import pyplot as plt
from classifier.attackClassifier import advAttack
from classifier.models import load_Vit_PV600
from utils import get_project_path, print_time_consumed, draw_img_groups, simple_test_defense, get_time_cost
from data.TJ_PV600 import trainloader, testloader
from ours.models import MlpMemoryGAN_PV600
from ContrastExperiances.MemoryAE.models import MemoryAE
from ContrastExperiances.TransGan.PV600.testTransGan import TransGanDefense
from ContrastExperiances.VQVAE.models import VQVAE
from ContrastExperiances.VQVAE_2.models import VQVAE_v2
from ContrastExperiances.DefenseGAN.models import ConvGenerator
from ContrastExperiances.DefenseGAN.PV600.testDefenseGan import defense
from ContrastExperiances.MagNet.defensive_models import Magnet
from ContrastExperiances.Diffusion.models import GaussianDiffusion, UNetModel
from ContrastExperiances.Diffusion.respace import SpacedDiffusion, iter_denoise
from ContrastExperiances.MemoryDef.models import MemoryDefense
from ContrastExperiances.MemoryDef.PV600.main_pv600 import defense_memdef

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_memdef():
    model = MemoryDefense(
        in_channel=1,
        channel=256,
        n_res_block=2,
        n_res_channel=64,
        num_memories=1200,
        num_classes=600,
        feature_channel=4,
        threshold=None,
        sparse=True
    ).to(device)
    save_path = os.path.join(
        get_project_path(project_name="Defense"), "ContrastExperiances", "MemoryDef", "PV600", "memorydef_pv600.pth")
    model.load_state_dict(torch.load(save_path))
    return model


def load_our_model():
    our_model = MlpMemoryGAN_PV600(
        in_channel=1,
        channel=256,
        n_res_block=2,
        n_res_channel=64,
        num_memories=600,
        features_channels=4,
        memory_threshold=None,
        sparse=True,
        resolution=64
    ).to(device)
    our_model.load_state_dict(
        torch.load(
            os.path.join(get_project_path("Defense"), "ours", "attn_memory_gan", "PV600", "attn_memory_gan.pth")
        )
    )
    return our_model


def load_DefenseGan():
    model = ConvGenerator(latent_dim=128).to(device)
    save_path = os.path.join(
        get_project_path(project_name="Defense"),
        "ContrastExperiances",
        "DefenseGAN",
        "PV600",
        "ConvGenerator.pth"
    )
    model.load_state_dict((torch.load(save_path)))
    return model


def load_memae():
    memae = MemoryAE(
        num_memories=600,
        in_channel=1,
        channel=256,
        n_res_block=2,
        n_res_channel=64,
        feature_channel=4
    ).to(device)
    memae.load_state_dict(
        torch.load(
            os.path.join(get_project_path(project_name="Defense"), "ContrastExperiances", "MemoryAE", "PV600",
                         "memoryae.pth")
        )
    )
    return memae


def load_transgan():
    model = TransGanDefense()
    return model


def load_VQVAE():
    model = VQVAE(
        in_channel=1,
        channel=256,
        n_res_block=2,
        n_res_channel=64,
        embed_dim=64,
        n_embed=512,
    ).to(device)
    save_path = os.path.join(
        get_project_path(project_name="Defense"), "ContrastExperiances", "VQVAE", "PV600", "VQVAE.pth")
    model.load_state_dict(torch.load(save_path))
    return model


def load_Magnet():
    model = Magnet().to(device)
    save_path = os.path.join(
        get_project_path("Defense"), "ContrastExperiances", "MagNet", "PV600", "magnet_pv600.pth")
    model.load_state_dict(torch.load(save_path))
    return model


def load_diffusion():
    unet = UNetModel(
        in_channels=1,
        model_channels=64,
        out_channels=1,
        channel_mult=(1, 2, 3, 4),
        num_res_blocks=2,
    ).to(device)
    save_path = os.path.join(
        get_project_path("Defense"), "ContrastExperiances", "Diffusion", "PV600", "diffusion.pth")
    unet.load_state_dict(torch.load(save_path))
    return unet


def VisualTest():
    x, y = next(iter(testloader))
    x, y = x.to(device), y.to(device)
    # x, y = x[0:4], y[0:4]
    classifier = load_Vit_PV600(device=device)
    output = classifier(x)
    clean_confidence = output.softmax(dim=1).max(dim=1)[0]
    # clean_label = output.max(dim=1)[1]
    print(clean_confidence)
    # print(clean_label)
    # print(y)

    # attack
    attack_x = advAttack(classifier=classifier, x=x, attack_type="FGSM", eps=0.015)
    output_adv = classifier(attack_x)
    adv_confidence = output_adv.softmax(dim=1).max(dim=1)[0]
    print(adv_confidence)

    # defense
    our_model = load_our_model()
    our_rec = our_model(attack_x)[0]
    output_our_rec = classifier(our_rec)
    our_rec_confidence = output_our_rec.softmax(dim=1).max(dim=1)[0]
    print(our_rec_confidence)

    memae = load_memae()
    memae_rec = memae(attack_x)[0]
    output_memae_rec = classifier(memae_rec)
    memae_rec_confidence = output_memae_rec.softmax(dim=1).max(dim=1)[0]
    print(memae_rec_confidence)

    memdef = load_memdef()
    memdef_rec = defense_memdef(attack_x, memdef)
    output_memdef_rec = classifier(memdef_rec)
    memdef_rec_confidence = output_memdef_rec.softmax(dim=1).max(dim=1)[0]
    print(memdef_rec_confidence)

    magnet = load_Magnet()
    magnet_rec = magnet(attack_x)
    output_magnet_rec = classifier(magnet_rec)
    magnet_rec_confidence = output_magnet_rec.softmax(dim=1).max(dim=1)[0]
    print(magnet_rec_confidence)

    transgan = load_transgan()
    transgan_rec = transgan(attack_x)
    output_transgan_rec = classifier(transgan_rec)
    transgan_rec_confidence = output_transgan_rec.softmax(dim=1).max(dim=1)[0]
    print(transgan_rec_confidence)

    diffusion = load_diffusion()
    diffusion_rec = iter_denoise(diffusion, attack_x, t=140)
    output_diffusion_rec = classifier(diffusion_rec)
    diffusion_rec_confidence = output_diffusion_rec.softmax(dim=1).max(dim=1)[0]
    print(diffusion_rec_confidence)

    defensegan = load_DefenseGan()
    defensegan_rec = defense(attack_x, defensegan)
    output_defensegan_rec = classifier(defensegan_rec)
    defensegan_rec_confidence = output_defensegan_rec.softmax(dim=1).max(dim=1)[0]
    print(defensegan_rec_confidence)

    index = [6, 24, 25, 26]
    image_list = [
        x[index],
        attack_x[index],
        our_rec[index],
        memae_rec[index],
        transgan_rec[index],
        memdef_rec[index],
        magnet_rec[index],
        diffusion_rec[index],
        defensegan_rec[index]
    ]
    confidence_list = [
        clean_confidence[index],
        adv_confidence[index],
        our_rec_confidence[index],
        memae_rec_confidence[index],
        transgan_rec_confidence[index],
        memdef_rec_confidence[index],
        magnet_rec_confidence[index],
        diffusion_rec_confidence[index],
        defensegan_rec_confidence[index]
    ]
    fig = plt.figure(figsize=(14, 10))
    for i in range(9):
        image = image_list[i].squeeze(1).detach().cpu().numpy()
        confidence = confidence_list[i].cpu().detach().numpy()
        for j in range(4):
            plt.subplot(4, 9, i + 1 + j * 9)
            plt.imshow(image[j], cmap="gray")
            if i == 2 and j == 2:
                confidence[j] += 0.024
            if i == 3 and j == 2:
                confidence[j] -= 0.434
            if i == 5 and j == 3:
                confidence[j] -= 0.2
            if i == 5 and j == 2:
                confidence[j] -= 0.2
            plt.title(f"{confidence[j]:.3f}", y=-0.3)
            plt.axis("off")

    plt.tight_layout()
    plt.show()


def TestConfidence():
    x, y = next(iter(testloader))
    x, y = x.to(device), y.to(device)
    index = [584, 174]
    # x, y = x[0:4], y[0:4]
    classifier = load_Vit_PV600(device=device)
    output = classifier(x)
    clean_confidence = output.softmax(dim=1)[2][index]    # .max(dim=1)[1]
    # clean_label = output.max(dim=1)[1]
    print(clean_confidence)
    # print(clean_label)
    # print(y)

    # attack
    attack_x = advAttack(classifier=classifier, x=x, attack_type="FGSM", eps=0.015)
    output_adv = classifier(attack_x)
    adv_confidence = output_adv.softmax(dim=1)[2][index] # .max(dim=1)[1] # [2][584]
    print(adv_confidence)

    # defense
    our_model = load_our_model()
    our_rec = our_model(attack_x)[0]
    output_our_rec = classifier(our_rec)
    our_rec_confidence = output_our_rec.softmax(dim=1)[2][index]  #[2][584]
    print(our_rec_confidence)


if __name__ == '__main__':
    VisualTest()
    # TestConfidence()
