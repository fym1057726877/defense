import torch
from torch import nn
from torch.nn import functional as F
from torch import einsum


class MemoryBlock(nn.Module):
    def __init__(self, num_memories, features, sparse=True, threshold=None, entropy_loss_coef=0.0002):
        super().__init__()

        self.num_memories = num_memories
        self.features = features

        self.memory = torch.zeros((self.num_memories, self.features))
        nn.init.kaiming_uniform_(self.memory)
        self.memory = nn.Parameter(self.memory)

        print(f"init memory with num_memories={self.num_memories}, features={self.features}")

        self.Cosine_Similiarity = nn.CosineSimilarity(dim=2)
        self.sparse = sparse
        if self.sparse:
            self.threshold = threshold or 1 / self.num_memories
            self.epsilon = 1e-12

        self.entropy_loss_coef = entropy_loss_coef

    def forward(self, x):
        B = x.shape[0]
        z = x.view(B, self.features).contiguous()

        ex_mem = self.memory.unsqueeze(0).repeat(B, 1, 1)  # [b, mem_dim, fea]
        ex_z = z.unsqueeze(1).repeat(1, self.num_memories, 1)  # [b, mem_dim, fea]

        mem_logit = self.Cosine_Similiarity(ex_z, ex_mem)  # [b, mem_dim]
        mem_weight = mem_logit.softmax(dim=1)  # [b, num_mem]

        # 稀疏寻址
        if self.sparse:
            mem_weight = ((F.relu(mem_weight - self.threshold) * mem_weight)
                          / (torch.abs(mem_weight - self.threshold) + self.epsilon))
            mem_weight = F.normalize(mem_weight, p=1, dim=1)

        z_hat = torch.matmul(mem_weight, self.memory).view(x.shape).contiguous()
        mem_loss = self.EntropyLoss(mem_weight)

        return z_hat, mem_loss

    def EntropyLoss(self, mem_weight):
        entropy_loss = -mem_weight * torch.log(mem_weight + 1e-12)
        entropy_loss = entropy_loss.sum() / mem_weight.shape[0]
        entropy_loss *= self.entropy_loss_coef
        return entropy_loss


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, kernel_size=3, stride=1, padding=1),
            ]
        else:
            raise NotImplemented("unkown stride")

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
            self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, kernel_size=3, stride=1, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(channel // 2, out_channel, kernel_size=4, stride=2, padding=1),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, kernel_size=4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class MultiHeadAttentionMemory(nn.Module):
    def __init__(self, num_memories, features, num_heads=16, sparse=True, threshold=None, entropy_loss_coef=0.0002):
        super().__init__()

        self.num_memories = num_memories
        self.features = features

        self.memory = torch.zeros((self.num_memories, self.features))
        nn.init.kaiming_uniform_(self.memory)
        self.memory = nn.Parameter(self.memory)

        self.attention = MultiHeadAttention(in_dims=features, num_heads=num_heads)
        print(f"init memory with num_memories={self.num_memories}, features={self.features}")

        self.Cosine_Similiarity = nn.CosineSimilarity(dim=2)
        self.sparse = sparse
        if self.sparse:
            self.threshold = threshold or 1 / self.num_memories
            self.epsilon = 1e-12

        self.entropy_loss_coef = entropy_loss_coef

    def forward(self, x):
        B, C, H, W = x.shape
        z = x.view(B, -1).unsqueeze(1).contiguous()

        z = self.attention(z).squeeze(1).contiguous()

        ex_mem = self.memory.unsqueeze(0).repeat(B, 1, 1)  # [b, mem_dim, fea]
        ex_z = z.unsqueeze(1).repeat(1, self.num_memories, 1)  # [b, mem_dim, fea]

        mem_logit = self.Cosine_Similiarity(ex_z, ex_mem)  # [b, mem_dim]
        mem_weight = mem_logit.softmax(dim=1)  # [b, num_mem]

        # 稀疏寻址
        if self.sparse:
            mem_weight = ((F.relu(mem_weight - self.threshold) * mem_weight)
                          / (torch.abs(mem_weight - self.threshold) + self.epsilon))
            mem_weight = F.normalize(mem_weight, p=1, dim=1)

        z_hat = torch.matmul(mem_weight, self.memory).view(x.shape).contiguous()
        mem_loss = self.EntropyLoss(mem_weight)

        return z_hat, mem_loss

    def EntropyLoss(self, mem_weight):
        entropy_loss = -mem_weight * torch.log(mem_weight + 1e-12)
        entropy_loss = entropy_loss.sum() / mem_weight.shape[0]
        entropy_loss *= self.entropy_loss_coef
        return entropy_loss


class MultiHeadAttention(nn.Module):
    def __init__(self, in_dims, out_dims=None, num_heads=1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.in_dims = in_dims

        out_dims = out_dims or in_dims
        self.out_dims = out_dims
        self.d_k = out_dims // num_heads

        self.query_linear = nn.Linear(in_dims, out_dims)
        self.key_linear = nn.Linear(in_dims, out_dims)
        self.value_linear = nn.Linear(in_dims, out_dims)
        self.output_linear = nn.Linear(out_dims, out_dims)

    def forward(self, input):
        batch_size = input.size(0)
        chanels = input.size(1)

        # Linear transformations
        query = self.query_linear(input)
        key = self.key_linear(input)
        value = self.value_linear(input)

        # Reshape for multi-head attention
        query = query.view(batch_size, chanels, self.num_heads, self.d_k)
        key = key.view(batch_size, chanels, self.num_heads, self.d_k)
        value = value.view(batch_size, chanels, self.num_heads, self.d_k)

        # Transpose dimensions for matrix multiplication
        query = query.transpose(1, 2)  # [batch_size, num_heads, chanels, d_k]
        key = key.transpose(1, 2)  # [batch_size, num_heads, chanels, d_k]
        value = value.transpose(1, 2)  # [batch_size, num_heads, chanels, d_k]

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1))  # [batch_size, num_heads, chanels, chanels]
        scores = scores / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        attention_weights = nn.Softmax(dim=-1)(scores)
        output = torch.matmul(attention_weights, value)  # [batch_size, num_heads, chanels, d_k]

        # Concatenate and reshape
        output = output.transpose(1, 2).contiguous().view(batch_size, chanels, self.out_dims)

        # Linear transformation for final output
        output = self.output_linear(output)

        return output


class MlpMemoryBlock(nn.Module):
    def __init__(self, num_memories, features, other_dims=1, sparse=True, threshold=None, entropy_loss_coef=0.0002):
        super().__init__()

        self.num_memories = num_memories
        self.features = features

        self.memory = torch.zeros((self.num_memories, self.features))
        nn.init.kaiming_uniform_(self.memory)
        self.memory = nn.Parameter(self.memory)

        print(f"init memory with num_memories={self.num_memories}, features={self.features}")

        self.weights = MlpforMemWeight(in_dims=self.features, other_dims=other_dims)
        self.sparse = sparse
        if self.sparse:
            self.threshold = threshold or 1 / self.num_memories
            self.epsilon = 1e-12

        self.entropy_loss_coef = entropy_loss_coef

    def forward(self, x):
        B = x.shape[0]
        z = x.view(B, self.features).contiguous()

        ex_mem = self.memory.unsqueeze(0).repeat(B, 1, 1)  # [b, mem_dim, fea]
        ex_z = z.unsqueeze(1).repeat(1, self.num_memories, 1)  # [b, mem_dim, fea]

        mem_logit = self.weights(ex_z, ex_mem)  # [b, mem_dim]
        mem_weight = mem_logit.softmax(dim=-1)  # [b, num_mem]

        # 稀疏寻址
        if self.sparse:
            mem_weight = ((F.relu(mem_weight - self.threshold) * mem_weight)
                          / (torch.abs(mem_weight - self.threshold) + self.epsilon))
            mem_weight = F.normalize(mem_weight, p=1, dim=1)

        z_hat = torch.matmul(mem_weight, self.memory).view(x.shape).contiguous()
        mem_loss = self.EntropyLoss(mem_weight)

        return z_hat, mem_loss

    def EntropyLoss(self, mem_weight):
        entropy_loss = -mem_weight * torch.log(mem_weight + 1e-12)
        entropy_loss = entropy_loss.sum() / mem_weight.shape[0]
        entropy_loss *= self.entropy_loss_coef
        return entropy_loss


class MlpforMemWeight(nn.Module):
    def __init__(self, in_dims, out_dims=None, other_dims=1):
        super().__init__()
        # out_dims = out_dims or in_dims
        # self.z_linear = nn.Linear(in_dims, out_dims)
        # self.memory_linear = nn.Linear(in_dims, out_dims)
        self.in_dims = in_dims
        # self.attention = nn.Linear(out_dims * 2, out_dims)
        self.attention = nn.Sequential(
            nn.Linear(self.in_dims * 2, self.in_dims),
            nn.ReLU(inplace=True),

            nn.Linear(self.in_dims, self.in_dims // 2),
            nn.ReLU(inplace=True),

            nn.Linear(self.in_dims // 2, self.in_dims // 4),
            nn.ReLU(inplace=True),

            nn.Linear(self.in_dims // 4, self.in_dims // 8),
            nn.Tanh()
        )
        self.other = nn.Parameter(torch.ones(self.in_dims // 8, other_dims, dtype=torch.float))

    def forward(self, z, memory):
        # z:(b,m,c)  memory:(b,m,c)
        # z = self.z_linear(z)  # z:(b,m,o)
        # memory = self.memory_linear(memory)  # memory:(b,m,o)
        weights = torch.cat([z, memory], dim=2)  # weights:(b,m,2c)
        weights = self.attention(weights)  # weights:(b,m,o)
        weights = einsum("bmo, oj -> bmj", weights, self.other)
        weights = torch.mean(weights, dim=-1).squeeze(-1)
        return weights


class MlpMemoryBlock_PV600(nn.Module):
    def __init__(self, num_memories, features, other_dims=1, sparse=True, threshold=None, entropy_loss_coef=0.0002):
        super().__init__()

        self.num_memories = num_memories
        self.features = features

        self.memory = torch.zeros((self.num_memories, self.features))
        nn.init.kaiming_uniform_(self.memory)
        self.memory = nn.Parameter(self.memory)

        print(f"init memory with num_memories={self.num_memories}, features={self.features}")

        self.weights = MlpforMemWeight_PV600(in_dims=self.features, other_dims=other_dims)
        self.sparse = sparse
        if self.sparse:
            self.threshold = threshold or 1 / self.num_memories
            self.epsilon = 1e-12

        self.entropy_loss_coef = entropy_loss_coef

    def forward(self, x):
        B = x.shape[0]
        z = x.view(B, self.features).contiguous()

        ex_mem = self.memory.unsqueeze(0).repeat(B, 1, 1)  # [b, mem_dim, fea]
        ex_z = z.unsqueeze(1).repeat(1, self.num_memories, 1)  # [b, mem_dim, fea]

        mem_logit = self.weights(ex_z, ex_mem)  # [b, mem_dim]
        mem_weight = mem_logit.softmax(dim=-1)  # [b, num_mem]

        # 稀疏寻址
        if self.sparse:
            mem_weight = ((F.relu(mem_weight - self.threshold) * mem_weight)
                          / (torch.abs(mem_weight - self.threshold) + self.epsilon))
            mem_weight = F.normalize(mem_weight, p=1, dim=1)

        z_hat = torch.matmul(mem_weight, self.memory).view(x.shape).contiguous()
        mem_loss = self.EntropyLoss(mem_weight)

        return z_hat, mem_loss

    def EntropyLoss(self, mem_weight):
        entropy_loss = -mem_weight * torch.log(mem_weight + 1e-12)
        entropy_loss = entropy_loss.sum() / mem_weight.shape[0]
        entropy_loss *= self.entropy_loss_coef
        return entropy_loss


class MlpforMemWeight_PV600(nn.Module):
    def __init__(self, in_dims, out_dims=None, other_dims=1):
        super().__init__()
        out_dims = out_dims or in_dims
        self.z_linear = nn.Linear(in_dims, out_dims)
        self.memory_linear = nn.Linear(in_dims, out_dims)
        self.in_dims = in_dims
        # self.attention = nn.Linear(out_dims * 2, out_dims)
        self.attention = nn.Linear(self.in_dims * 2, self.in_dims)
        self.other = nn.Parameter(torch.ones(self.in_dims, other_dims, dtype=torch.float))
        # nn.init.xavier_uniform_(self.other.data)
        self.tanh = nn.Tanh()

    def forward(self, z, memory):
        # z:(b,m,c)  memory:(b,m,c)
        z = self.z_linear(z)  # z:(b,m,o)
        memory = self.memory_linear(memory)  # memory:(b,m,o)
        weights = torch.cat([z, memory], dim=2)  # weights:(b,m,2c)
        weights = self.attention(weights)  # weights:(b,m,o)
        weights = self.tanh(weights)
        weights = einsum("bmo, oj -> bmj", weights, self.other)
        weights = torch.mean(weights, dim=-1).squeeze(-1)
        return weights


def test():
    model = MlpforMemWeight(in_dims=256)
    x = torch.randn(4, 5, 256)
    y = torch.randn(4, 5, 256)
    out = model(x, y)
    print(out, out.shape)


# test()
