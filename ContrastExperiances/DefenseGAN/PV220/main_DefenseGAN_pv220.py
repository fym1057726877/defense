import os
import torch
from torch import optim
from torch.autograd import Variable
from ContrastExperiances.DefenseGAN.models import ConvGenerator, ConvDiscriminator
from data.VERA_PV220 import trainloader, testloader
from utils import get_project_path, draw_img_groups
from tqdm import tqdm


class TrainConvGAN:
    def __init__(self):
        super(TrainConvGAN, self).__init__()
        self.device = "cuda"
        self.latent_dim = 256
        self.dis_iters = 4
        self.g_lr = 1e-5
        self.d_lr = 1e-5
        self.generator = ConvGenerator(self.latent_dim).to(self.device)
        self.discriminator = ConvDiscriminator().to(self.device)

        self.train_loader = trainloader
        self.test_loader = testloader

        self.gen_path = os.path.join(
            get_project_path("Defense"),
            "ContrastExperiances",
            "DefenseGAN",
            "PV220",
            "ConvGenerator.pth"
        )
        self.dis_path = os.path.join(
            get_project_path("Defense"),
            "ContrastExperiances",
            "DefenseGAN",
            "PV220",
            "ConvDiscriminator.pth"
        )

        self.generator.load_state_dict(torch.load(self.gen_path))
        self.discriminator.load_state_dict(torch.load(self.dis_path))

    def train(self, epochs=100):
        optim_g = optim.Adam(self.generator.parameters(), lr=self.g_lr, weight_decay=1e-3, betas=(0.5, 0.99))
        optim_d = optim.Adam(self.discriminator.parameters(), lr=self.d_lr, weight_decay=1e-3, betas=(0.5, 0.99))
        lr_scheduler_g = optim.lr_scheduler.StepLR(optim_g, step_size=10, gamma=0.9)
        lr_scheduler_d = optim.lr_scheduler.StepLR(optim_d, step_size=10, gamma=0.9)
        batch_count = len(self.train_loader)
        for epoch in range(epochs):
            g_epoch_loss, d_epoch_loss = 0, 0
            indice = tqdm(enumerate(self.train_loader), desc=f"train {epoch+1}/{epochs}", total=batch_count)
            for i, (img, label) in indice:
                img = img.to(self.device)

                z = torch.randn((img.size(0), self.latent_dim)).to(self.device)

                self.generator.eval()

                fakeImg = self.generator(z)

                self.discriminator.train()
                optim_d.zero_grad()

                gp = self.gradient_penalty(img, fakeImg, self.device)
                loss_d = -torch.mean(self.discriminator(img)) + torch.mean(self.discriminator(fakeImg)) + gp
                # gp.backward(retain_graph=True)
                loss_d.backward()
                optim_d.step()

                d_epoch_loss += loss_d

                if i % self.dis_iters == 0:
                    self.generator.train()
                    self.discriminator.eval()
                    optim_g.zero_grad()
                    fake_img = self.generator(z)
                    loss_g = -torch.mean(self.discriminator(fake_img))
                    loss_g.backward()
                    optim_g.step()

                    g_epoch_loss += loss_g

            d_epoch_loss /= batch_count
            g_epoch_loss /= (batch_count / self.dis_iters)
            print(f"[G loss: {g_epoch_loss:.6f}] [D loss: {d_epoch_loss:.6f}]")
            if epoch % 500 == 0 and epoch != 0:
                self.test()
            if optim_g.state_dict()['param_groups'][0]['lr'] > self.g_lr / 1e2:
                lr_scheduler_g.step()
                lr_scheduler_d.step()

            self.save_model()

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

    def save_model(self):
        torch.save(self.generator.state_dict(), self.gen_path)
        torch.save(self.discriminator.state_dict(), self.dis_path)

    def test(self):
        test_input = torch.randn(size=(10, self.latent_dim), device=self.device)
        x, _ = next(iter(self.test_loader))
        gen_imgs = self.generator(test_input)
        draw_img_groups(img_groups=[x, gen_imgs])


def trainConvGANmain():
    train_Conv_GAN = TrainConvGAN()
    train_Conv_GAN.train(10000)
    # train_Conv_GAN.test()


if __name__ == "__main__":
    trainConvGANmain()
