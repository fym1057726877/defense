import torch
import torch.nn as nn
from torch import optim


def GradientDescentReconstruct(img, generator, device="cpu", L=200, R=10, lr=0.015, latent_dim=256):
    B = img.size(0)
    img = img.repeat(R, 1, 1, 1).to(device)
    z = torch.randn((B * R, latent_dim)).to(device)
    z.requires_grad = True
    optim_g = optim.RMSprop([z], lr=lr)
    lr_scheduler_g = optim.lr_scheduler.StepLR(optim_g, step_size=5, gamma=0.99)
    loss_fun1 = nn.MSELoss()

    for iterIndex in range(L):
        fake_img = generator(z)
        optim_g.zero_grad()
        loss_g = loss_fun1(fake_img, img)
        loss_g.backward()
        optim_g.step()
        if optim_g.state_dict()['param_groups'][0]['lr'] * 1000 > 1:
            lr_scheduler_g.step()

    fake_img = generator(z)
    distance = torch.mean(((fake_img - img) ** 2).view(B * R, -1), dim=1)
    rec_img = None
    distance = distance.view(R, B).transpose(1, 0)
    for imgIndex in range(B):
        j = torch.argmin(distance[imgIndex], dim=0)
        index = j * B + imgIndex
        current_rec_img = fake_img[index].unsqueeze(0)
        if rec_img is None:
            rec_img = current_rec_img
        else:
            rec_img = torch.cat((rec_img, current_rec_img), dim=0)
    return rec_img.to(device)

