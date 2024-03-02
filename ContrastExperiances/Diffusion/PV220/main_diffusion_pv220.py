
import os
import torch
import logging
from torch import optim
from ContrastExperiances.Diffusion.models import GaussianDiffusion, UNetModel
from ContrastExperiances.Diffusion.respace import SpacedDiffusion, iter_denoise
from data.VERA_PV220 import trainloader, testloader
from time import time
from tqdm import tqdm
from utils import simple_test_defense, draw_img_groups, get_project_path


class TrainDiffusion:
    def __init__(
            self,
            lr=2e-4,
    ):
        super(TrainDiffusion, self).__init__()
        self.device = "cuda"
        self.lr = lr
        self.diffsuion = GaussianDiffusion()
        self.unet = UNetModel(
            in_channels=1,
            model_channels=64,
            out_channels=1,
            channel_mult=(1, 2, 3, 4),
            num_res_blocks=2,
        ).to(self.device)
        self.save_path = os.path.join(get_project_path("Defense"), "ContrastExperiances", "Diffusion", "PV220", "diffusion.pth")
        self.unet.load_state_dict(torch.load(self.save_path))

        self.trainloader = trainloader
        self.testloader = testloader

        # self.classifier_name = "ModelB_PV600"
        # self.classifier_name = "Vit_PV600"
        self.classifier_name = "SwinTransformer_PV600"

    def train(self, epochs):
        start = time()
        optimizer = optim.AdamW(self.unet.parameters(), lr=self.lr, weight_decay=0.005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)
        for epoch in range(epochs):
            count = len(self.trainloader)
            epoch_loss = 0
            self.unet.train()
            for step, (img, _) in tqdm(enumerate(self.trainloader), desc=f"train step {epoch+1}/{epochs}", total=count):
                optimizer.zero_grad()
                img = img.to(self.device)
                batch_size = img.shape[0]

                t = self.diffsuion.get_rand_t(batch_size, self.device, min_t=0, max_t=1000)

                loss = self.diffsuion.training_losses(model=self.unet, x_start=img, t=t)
                epoch_loss += loss
                loss.backward()
                optimizer.step()

            if optimizer.state_dict()['param_groups'][0]['lr'] > self.lr / 1e2:
                scheduler.step()

            torch.save(self.unet.state_dict(), self.save_path)

            print(f"Epoch:{epoch+1}/{epochs}  Loss:{epoch_loss:.8f}")

            if epoch % 20 == 0 and epoch != 0:
                self.test_ddim()

        end = time()
        seconds = int(end - start)
        minutes = seconds // 60
        remain_second = seconds % 60
        print(f"time consumed: {minutes}min{remain_second}s")

    def test_ddim(self):
        self.unet.eval()
        spacediffusion = SpacedDiffusion(num_ddpm_timesteps=1000, num_ddim_timesteps=50)
        imgs, _ = next(iter(self.testloader))
        imgs = imgs.to(self.device)

        torch.manual_seed(10)
        noise = torch.randn_like(imgs)

        final_sample = spacediffusion.ddim_sample_loop(self.unet, shape=imgs.shape, noise=noise, progress=True)

        draw_img_groups([imgs, final_sample])

    def test_ddpm(self):
        self.unet.eval()
        imgs, _ = next(iter(self.testloader))
        imgs = imgs.to(self.device)
        noise = torch.randn_like(imgs)
        final_sample = self.diffsuion.p_sample_loop(self.unet, shape=imgs.shape, noise=noise, progress=True)
        draw_img_groups([noise, final_sample])

    def test_iter_denoise(self, t=50):
        assert t % 10 == 0
        self.unet.eval()
        imgs, _ = next(iter(self.testloader))
        imgs = imgs.to(self.device)
        final_sample = iter_denoise(self.unet, imgs=imgs, t=t)
        draw_img_groups([imgs, final_sample])

    def test_defense_one(self):
        classifier_name = "SwinTransformer_PV600"
        log = simple_test_defense(defense_model=self.unet, attack_type="FGSM", eps=0.015,
                                  testloader=self.testloader, defense_fn=defense,
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

        log1 = simple_test_defense(defense_model=self.unet, attack_type="FGSM", eps=0.01,
                                   testloader=self.testloader, defense_fn=defense,
                                   classifier_name=classifier_name, device=self.device)
        print(log1)
        log2 = simple_test_defense(defense_model=self.unet, attack_type="FGSM", eps=0.03,
                                   testloader=self.testloader, defense_fn=defense,
                                   classifier_name=classifier_name, device=self.device)
        print(log2)
        log3 = simple_test_defense(defense_model=self.unet, attack_type="PGD", eps=0.01,
                                   testloader=self.testloader, defense_fn=defense,
                                   classifier_name=classifier_name, device=self.device)
        print(log3)
        log4 = simple_test_defense(defense_model=self.unet, attack_type="PGD", eps=0.015,
                                   testloader=self.testloader, defense_fn=defense,
                                   classifier_name=classifier_name, device=self.device)
        print(log4)

    def total_test(self):
        torch.manual_seed(10)
        logging.basicConfig(filename="./logs/diffusion_test_pv220.logs", filemode="w", level=logging.INFO, format="")
        classifier_name_list = [
            # "FV_CNN_PV220",
            # "PV_CNN_PV220",
            # "LightweightFVCNN_PV220",
            # "FVRAS_Net_PV220",
            # "ModelB_PV220",
            "Res2Net_PV220",
            "Vit_PV220",
            "SwinTransformer_PV220",
        ]
        for classifier_name in classifier_name_list:
            log1 = simple_test_defense(defense_model=self.unet, defense_fn=defense, attack_type="FGSM", eps=0.01,
                                       classifier_name=classifier_name, progress=False, device=self.device)
            print(log1)
            logging.info(log1)
            log2 = simple_test_defense(defense_model=self.unet, defense_fn=defense, attack_type="FGSM", eps=0.015,
                                       classifier_name=classifier_name, progress=False, device=self.device)
            print(log2)
            logging.info(log2)
            log3 = simple_test_defense(defense_model=self.unet, defense_fn=defense, attack_type="PGD", eps=0.01,
                                       classifier_name=classifier_name, progress=False, device=self.device)
            print(log3)
            logging.info(log3)
            log4 = simple_test_defense(defense_model=self.unet, defense_fn=defense, attack_type="PGD", eps=0.015,
                                       classifier_name=classifier_name, progress=False, device=self.device)
            print(log4)
            logging.info(log4)


def defense(x, unet):
    rec = iter_denoise(unet, imgs=x, t=100)
    return rec


def main():
    # trainer = TrainDiffusion(unet_pred=ModelMeanType.START_X)
    trainer = TrainDiffusion()
    # trainer.train(2000)
    # trainer.test_ddim()
    # trainer.test_ddpm()
    # trainer.test_iter_denoise(t=100)
    # trainer.test_defense()
    trainer.total_test()
    # trainer.test_defense_one()


if __name__ == "__main__":
    main()



