import time
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.torch.attacks.hop_skip_jump_attack import hop_skip_jump_attack
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.torch.attacks.spsa import spsa
from torchattacks.attacks.square import Square
from torchattacks.attacks.pixle import Pixle
from advertorch.attacks import SinglePixelAttack, LocalSearchAttack
from foolbox import PyTorchModel
from foolbox.attacks.boundary_attack import BoundaryAttack
import copy


def advAttack(classifier, x, attack_type, eps, iters=30, y=None):
    if attack_type == "RandFGSM":
        alpha = 0.005
        x = x + alpha * torch.sign(torch.randn(x.shape).to(x.device))
        # x = torch.clip(x + alpha * torch.sign(torch.randn(x.shape).to(device)), 0, 1)
        eps2 = eps - alpha
        x_adv = fast_gradient_method(classifier, x, eps2, np.inf)
    elif attack_type == "FGSM":
        x_adv = fast_gradient_method(classifier, x, eps, np.inf)
    elif attack_type == "PGD":
        x_adv = projected_gradient_descent(classifier, x, eps=eps, eps_iter=1 / 255,
                                           nb_iter=min(255 * eps + 4, 1.25 * (eps * 255)), norm=np.inf)
    elif attack_type == "HSJA":
        x_adv = hop_skip_jump_attack(classifier, x, norm=2, initial_num_evals=0, max_num_evals=50,
                                     batch_size=x.shape[0], verbose=False, num_iterations=iters)
    elif attack_type == "CW":
        x_adv = carlini_wagner_l2(classifier, x, n_classes=600, clip_min=x.min(), clip_max=x.max(),
                                  max_iterations=iters)
    elif attack_type == "SPSA":
        x_adv = spsa(classifier, x, y=y, eps=eps, norm=np.inf, nb_iter=iters,
                     spsa_samples=x.shape[0], spsa_iters=10, learning_rate=0.5, delta=0.5)
    else:
        raise RuntimeError(f"The attack type {attack_type} is invalid!")
    del classifier
    return x_adv


def no_Softmax(model):
    c_model = copy.deepcopy(model)
    if isinstance(list(model.children())[-1], nn.Softmax):
        c_model.softmax = nn.Identity()
    return c_model


def generateAdvImage(
        classifier,
        attack_dataloader,
        savepath=None,
        attack_type="FGSM",
        eps=0.03,
        iters=30,
        device="cuda",
        progress=False,
        logits=True
):
    print(f"-------------------------------------------------\n"
          f"Generating Adversarial Examples ...\n"
          f"eps = {eps} attack = {attack_type}")
    time.sleep(1)

    def accuracy(y, y1):
        return (y.max(dim=1)[1] == y1).sum()

    dataloader = attack_dataloader
    train_acc, adv_acc, train_n = 0, 0, 0
    normal_data, adv_data, label_data = None, None, None
    logits_model = no_Softmax(classifier)
    attack_model = logits_model if logits else classifier
    if progress:
        indice = tqdm(enumerate(dataloader), total=len(dataloader))
    else:
        indice = enumerate(dataloader)
    for index, (x, label) in indice:
        x, label = x.to(device), label.to(device)

        with torch.no_grad():
            pred = classifier(x)
            train_acc += accuracy(pred, label)

        x_adv = advAttack(classifier=attack_model, x=x, attack_type=attack_type, eps=eps, iters=iters, y=label)

        with torch.no_grad():
            y_adv = classifier(x_adv)
            adv_acc += accuracy(y_adv, label)

            train_n += label.size(0)

            x, x_adv, label = x.data, x_adv.data, label.data
            if normal_data is None:
                normal_data, adv_data, label_data = x, x_adv, label
            else:
                normal_data = torch.cat((normal_data, x))
                adv_data = torch.cat((adv_data, x_adv))
                label_data = torch.cat((label_data, label))

    print(f"Accuracy(normal) {torch.true_divide(train_acc, train_n):.6f}\n"
          f"Accuracy({attack_type}) {torch.true_divide(adv_acc, train_n):.6f}\n"
          f"-------------------------------------------------")

    adv_data = {"normal": normal_data, "adv": adv_data, "label": label_data}
    torch.save(adv_data, savepath)
    del logits_model
    return adv_data


def generateAdvData_for_memdef(
        classifier,
        attack_dataloader,
        attack_type="PGD",
        eps=0.005,
        iters=30,
        device="cuda",
        progress=False,
        logits=True,
        batch_size=50,
        shuffle=True
):
    print(f"-------------------------------------------------\n"
          f"Generating Adversarial Examples ...\n"
          f"eps = {eps} attack = {attack_type}")
    time.sleep(1)

    def accuracy(y, y1):
        return (y.max(dim=1)[1] == y1).sum()

    dataloader = attack_dataloader
    train_acc, adv_acc, train_n = 0, 0, 0
    normal_data, adv_data, label_data = None, None, None
    logits_model = no_Softmax(classifier)
    attack_model = logits_model if logits else classifier
    if progress:
        indice = tqdm(enumerate(dataloader), total=len(dataloader))
    else:
        indice = enumerate(dataloader)
    for index, (x, label) in indice:
        x, label = x.to(device), label.to(device)

        with torch.no_grad():
            pred = classifier(x)
            train_acc += accuracy(pred, label)

        x_adv = advAttack(classifier=attack_model, x=x, attack_type=attack_type, eps=eps, iters=iters, y=label)

        with torch.no_grad():
            y_adv = classifier(x_adv)
            adv_acc += accuracy(y_adv, label)

            train_n += label.size(0)

            x, x_adv, label = x.data, x_adv.data, label.data
            if normal_data is None:
                normal_data, adv_data, label_data = x, x_adv, label
            else:
                normal_data = torch.cat((normal_data, x))
                adv_data = torch.cat((adv_data, x_adv))
                label_data = torch.cat((label_data, label))

    print(f"Accuracy(normal) {torch.true_divide(train_acc, train_n):.6f}\n"
          f"Accuracy({attack_type}) {torch.true_divide(adv_acc, train_n):.6f}\n"
          f"-------------------------------------------------")

    from torch.utils.data import DataLoader, TensorDataset
    dataloder = DataLoader(TensorDataset(adv_data, label_data), batch_size=batch_size, shuffle=shuffle)
    del logits_model
    return dataloder

