import os
import torch
from torch import optim
from torch.autograd import Variable
from data.FV500 import trainloader
from ContrastExperiances.TransGan.TransGanModel import SwinTransGenerator, SwinTransDiscriminator
from utils import get_project_path, draw_img_groups
from tqdm import tqdm


# dataset_name = "Handvein"
dataset_name = "Handvein3"
# dataset_name = "Fingervein2"

g_embed_dim = 512
d_embed_dim = 768

# g_depths = [2, 4, 2, 2]
g_depths = [4, 2, 2, 2]
# d_depths = [2, 2, 6, 2]
d_depths = [4, 2, 2, 2]

# is_local = True
is_local = False
is_peg = True
# is_peg = False

if dataset_name == "Handvein" or dataset_name == 'Handvein3':
    width = 64
    height = 64
    bottom_width = 8
    bottom_height = 8
    window_size = 4

if dataset_name == 'Fingervein2':
    width = 128
    height = 64
    bottom_width = 16
    bottom_height = 8
    window_size = 4


class TrainTransGAN:
    def __init__(self):
        super(TrainTransGAN, self).__init__()
        self.train_data = trainloader
        self.latent_dim = 256

        self.g_lr = 1e-5
        self.d_lr = 1e-5
        self.device = "cuda"
        self.delay = 3
        self.generator = SwinTransGenerator(
            embed_dim=g_embed_dim,
            bottom_width=bottom_width,
            bottom_height=bottom_height,
            window_size=window_size,
            depth=g_depths,
            is_local=is_local,
            is_peg=is_peg
        ).to(self.device)
        self.discriminator = SwinTransDiscriminator(
            img_height=height,
            img_width=width,
            patch_size=window_size,
            embed_dim=d_embed_dim,
            depth=d_depths,
            is_local=is_local,
            is_peg=is_peg
        ).to(self.device)

        self.gen_path = os.path.join(
            get_project_path(project_name="Defense"),
            "ContrastExperiances",
            "TransGan",
            "FV500",
            "TransGenerator.pth"
        )
        self.dis_path = os.path.join(
            get_project_path(project_name="Defense"),
            "ContrastExperiances",
            "TransGan",
            "FV500",
            "TransDiscriminator.pth"
        )

        self.generator.load_state_dict(torch.load(self.gen_path))
        self.discriminator.load_state_dict(torch.load(self.dis_path))

    def train(self, epochs):
        optim_g = optim.Adam(self.generator.parameters(), lr=self.g_lr, weight_decay=1e-3, betas=(0.0, 0.99))
        optim_d = optim.Adam(self.discriminator.parameters(), lr=self.d_lr, weight_decay=1e-3, betas=(0.0, 0.99))
        batch_count = len(self.train_data)
        for epoch in range(epochs):
            g_epoch_loss, d_epoch_loss = 0, 0
            iter_object = tqdm(enumerate(self.train_data), desc=f"train step {epoch+1}/{epochs}", total=batch_count)
            for i, (img, label) in iter_object:
                img = img.to(self.device)

                z = torch.randn((img.size(0), self.latent_dim)).to(self.device)

                self.generator.eval()
                self.discriminator.train()

                optim_d.zero_grad()
                fake_img = self.generator(z)

                gp = self.gradient_penalty(img, fake_img, self.device)
                loss_d = -torch.mean(self.discriminator(img)) + torch.mean(self.discriminator(fake_img)) + gp
                gp.backward(retain_graph=True)
                loss_d.backward()
                optim_d.step()

                d_epoch_loss += loss_d

                if i % self.delay == 0:
                    self.generator.train()
                    self.discriminator.eval()
                    optim_g.zero_grad()
                    fake_img = self.generator(z)
                    loss_g = -torch.mean(self.discriminator(fake_img))
                    loss_g.backward()
                    optim_g.step()
                    g_epoch_loss += loss_g

            if (epoch+1) % 50 == 0:
                self.test()

            print(f"G_loss:{g_epoch_loss / (batch_count / self.delay):.6f}  D_Loss:{d_epoch_loss / batch_count:.6f}")
            self.save_model()

    def save_model(self):
        torch.save(self.generator.state_dict(), self.gen_path)
        torch.save(self.discriminator.state_dict(), self.dis_path)

    def gradient_penalty(self, img, fake_img, device):
        B = img.size(0)
        eps = torch.randn((B, 1, 1, 1)).to(device)
        x_inter = (eps * img + (1 - eps) * fake_img).requires_grad_(True).to(device)
        d_x_inter = self.discriminator(x_inter).to(device)
        grad_tensor = Variable(torch.Tensor(B, 1).fill_(1.0), requires_grad=False).to(device)
        grad = torch.autograd.grad(
            outputs=d_x_inter,
            inputs=x_inter,
            grad_outputs=grad_tensor,
            retain_graph=True,
            create_graph=True,
            only_inputs=True
        )[0]
        grad = grad.reshape(B, -1)
        gradient_penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean()
        return 10 * gradient_penalty

    def test(self):
        test_input = torch.randn(size=(40, self.latent_dim), device=self.device)
        gen_imgs = self.generator(test_input)
        # print(gen_imgs[0] * 100)
        draw_img_groups([gen_imgs])


def trainTransGAN():
    # seed = 1  # seed for random function
    worker = TrainTransGAN()
    worker.train(8000)
    # worker.test()


if __name__ == '__main__':
    trainTransGAN()
