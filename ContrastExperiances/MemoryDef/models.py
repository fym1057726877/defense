import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from ours.modules import Encoder, Decoder


class MemoryDefense(nn.Module):
    def __init__(
            self,
            num_memories=600,
            in_channel=1,
            channel=256,
            n_res_block=2,
            n_res_channel=64,
            feature_channel=4,
            num_classes=600,
            threshold=None,
            sparse=True
    ):
        super(MemoryDefense, self).__init__()
        assert num_memories % num_classes == 0
        feature_size = feature_channel * 16 * 16

        # Encoder
        self.encoder = Encoder(in_channel=in_channel, channel=channel, n_res_channel=n_res_channel,
                               n_res_block=n_res_block, stride=4)

        self.pre_conv = nn.Conv2d(channel, feature_channel, 3, 1, 1)
        self.post_conv = nn.Conv2d(feature_channel, channel, 3, 1, 1)

        self.memory = MemoryBlock(num_memories=num_memories, feature_size=feature_size, num_classes=num_classes,
                                  threshold=threshold, sparse=sparse)

        # Decoder
        self.decoder = Decoder(in_channel=channel, out_channel=in_channel, channel=channel,
                               n_res_channel=n_res_channel, n_res_block=n_res_block, stride=4)

        self.beta = 1e-4
        self.entropy_loss_coef = 2e-4

    def forward(self, x, lab):
        # Encoder
        x = self.encoder(x)
        z = self.pre_conv(x)
        z_hat_target, z_hat_non_target, mem_weight, mem_weight_hat = self.memory(z, lab)

        # Decoder
        rec_x = self.decoder(self.post_conv(z_hat_target))
        rec_x_hat = self.decoder(self.post_conv(z_hat_non_target))

        return dict(rec_x=rec_x, rec_x_hat=rec_x_hat,
                    mem_weight=mem_weight, mem_weight_hat=mem_weight_hat)

    def config_optim(self, lr=3e-4):
        return optim.Adam(list(self.encoder.parameters()) +
                          list(self.memory.parameters()) +
                          list(self.decoder.parameters()) +
                          list(self.pre_conv.parameters()) +
                          list(self.post_conv.parameters()),
                          lr=lr, betas=(0.9, 0.999))

    def training_losses(self, x, label):
        out = self(x, label)
        rec_x, mem_weight = out["rec_x"], out["mem_weight"]
        rec_x_hat, mem_weight_hat = out["rec_x_hat"], out["mem_weight_hat"]
        loss_target = F.mse_loss(rec_x, x) + self.EntropyLoss(mem_weight)
        loss_non_target = F.mse_loss(rec_x_hat, x) + self.EntropyLoss(mem_weight_hat)
        loss = loss_target - self.beta * loss_non_target.sigmoid()
        return loss

    def EntropyLoss(self, mem_weight):
        entropy_loss = -mem_weight * torch.log(mem_weight + 1e-12)
        entropy_loss = entropy_loss.sum() / mem_weight.shape[0]
        entropy_loss *= self.entropy_loss_coef
        return entropy_loss


class MemoryBlock(nn.Module):
    def __init__(self, num_memories=600, feature_size=None, threshold=None, num_classes=600, sparse=True):
        super().__init__()

        self.num_memories = num_memories
        self.num_classes = num_classes
        self.feature_size = feature_size
        assert self.feature_size is not None

        # Memory
        init_mem = torch.zeros(self.num_memories, self.feature_size)
        nn.init.kaiming_uniform_(init_mem)
        self.memory = nn.Parameter(init_mem)

        self.cosine_similarity = nn.CosineSimilarity(dim=2)

        self.hardshrink = nn.Hardshrink(lambd=1e-12)  # 1e-12
        self.threshold = threshold or 1 / self.num_memories
        self.sparse = sparse

    def forward(self, z, lab):
        batch, c, h, w = z.shape
        z = z.view(batch, -1)
        assert z.shape[1] == self.feature_size

        # Memory
        ex_mem = self.memory.unsqueeze(0).repeat(batch, 1, 1)
        ex_z = z.unsqueeze(1).repeat(1, self.num_memories, 1)
        mem_logit = self.cosine_similarity(ex_z, ex_mem)
        mem_weight = F.softmax(mem_logit, dim=1)

        # Masking using one hot encoding scheme over memory slots.
        m1, m2 = self.masking(lab)  # Generating Mask
        masked_mem_weight = mem_weight * m1  # Masking target class
        masked_mem_weight_hat = mem_weight * m2  # Masking non-target class

        if self.sparse:
            z_hat_target = torch.mm(masked_mem_weight, self.memory)  # M
            z_hat_non_target = torch.mm(masked_mem_weight_hat, self.memory)  # Matrix Multiplication: Att_W x Mem
        else:
            # Unmask Weight Target Class
            masked_mem_weight = self.hardshrink(masked_mem_weight)
            masked_mem_weight = masked_mem_weight / masked_mem_weight.norm(p=1, dim=1).unsqueeze(1).expand(
                batch, self.num_memories)
            z_hat_target = torch.mm(masked_mem_weight, self.memory)
            # Mask Weight Non-target Class
            masked_mem_weight_hat = self.hardshrink(masked_mem_weight_hat)
            masked_mem_weight_hat = masked_mem_weight_hat / masked_mem_weight_hat.norm(p=1, dim=1).unsqueeze(1).expand(
                batch, self.num_memories)
            z_hat_non_target = torch.mm(masked_mem_weight_hat, self.memory)

        z_hat_target, z_hat_non_target = z_hat_target.view(batch, c, h, w).contiguous(), z_hat_non_target.view(batch, c, h, w).contiguous()
        return z_hat_target, z_hat_non_target, masked_mem_weight, masked_mem_weight_hat

    def masking(self, label):
        memoryPerClass = self.num_memories // self.num_classes

        batch_size = len(label)

        mask1 = torch.zeros(batch_size, self.num_memories)
        mask2 = torch.ones(batch_size, self.num_memories)
        ones = torch.ones(memoryPerClass)
        zeros = torch.zeros(memoryPerClass)

        for i in range(batch_size):
            lab = torch.arange(memoryPerClass * label[i], memoryPerClass * (label[i] + 1), dtype=torch.long)
            if lab.nelement() == 0:
                print("Label tensor empty in the memory module.")
            else:
                mask1[i, lab] = ones
                mask2[i, lab] = zeros
        return mask1.to(label.device), mask2.to(label.device)
