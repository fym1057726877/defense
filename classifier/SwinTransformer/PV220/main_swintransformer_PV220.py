import os
import torch
from torch import optim, nn
from tqdm import tqdm
from timm.models.swin_transformer_v2 import SwinTransformerV2
from utils import get_project_path, accuracy
from data.VERA_PV220 import trainloader, testloader


class TrainClassifier:
    def __init__(
            self,
            lr=2e-4,
    ):
        super(TrainClassifier, self).__init__()
        self.device = "cuda"
        model_name = "SwinTransformer_PV220"
        self.classifier = SwinTransformerV2(img_size=64, in_chans=1, num_classes=220, num_heads=[2, 4, 6, 8],
                                            patch_size=4, embed_dim=96, depths=[2, 2, 4, 2], window_size=4,
                                            mlp_ratio=4).to(self.device)

        self.train_loader, self.test_loader = trainloader, testloader

        self.save_path = os.path.join(get_project_path("Defense"), "classifier", "SwinTransformer", "PV220",
                                      f"{model_name}.pth")
        # self.classifier.load_state_dict(torch.load(self.save_path))

        # loss function
        self.loss_fun = nn.CrossEntropyLoss()
        self.lr = lr

        self.optimer = optim.AdamW(self.classifier.parameters(), lr=self.lr, weight_decay=0.05)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimer, step_size=10, gamma=0.99)

    def train(self, epochs, progress=False):
        best_acc = 0
        for e in range(epochs):
            correct_num = 0
            train_num = 0
            epoch_loss = 0
            self.classifier.train()
            # start train
            count = len(self.train_loader)
            if progress:
                iterobject = tqdm(enumerate(self.train_loader), desc=f"train {e + 1}/{epochs}", total=batch_count)
            else:
                iterobject = enumerate(self.train_loader)
            for _, (img, label) in iterobject:
                self.optimer.zero_grad()
                img, label = img.to(self.device), label.to(self.device)
                pred = self.classifier(img)
                loss = self.loss_fun(pred, label.long())
                loss.backward()
                self.optimer.step()

                correct_num += accuracy(pred, label)
                train_num += label.size(0)
                epoch_loss += loss

            train_acc = correct_num / train_num
            test_acc = self.evaluate(show=False, progress=False)
            epoch_loss /= count

            if test_acc > best_acc:
                best_acc = test_acc
                print("saving the best model")
                torch.save(self.classifier.state_dict(), self.save_path)
            print(
                f"Epoch {e}/{epochs}   Loss:{epoch_loss:.8f}   "
                f"Train_Acc: {train_acc:.3f}   Test_Acc: {test_acc:.3f}   Best_Test_Acc: {best_acc:.3f}"
            )

    def evaluate(self, show=True, progress=False):
        correct_num, eval_num = 0, 0
        self.classifier.eval()
        if progress:
            indice = tqdm(enumerate(self.test_loader), desc="test step", total=len(self.test_loader))
        else:
            indice = enumerate(self.test_loader)
        for index, (x, label) in indice:
            x, label = x.to(self.device), label.to(self.device)
            pred = self.classifier(x)
            correct_num += accuracy(pred, label)
            eval_num += label.size(0)
        acc = correct_num / eval_num
        if show:
            print(f"acc:{acc:.6f}")
        return acc


def main_Cls():
    # seed = 1  # the seed for random function
    train_Classifier = TrainClassifier()
    train_Classifier.train(500)
    # train_Classifier.evaluate()


if __name__ == "__main__":
    main_Cls()
