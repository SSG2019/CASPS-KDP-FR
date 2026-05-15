import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleQAMChannel:
    """最小 16-QAM 信道示例，用于公开代码中的单点复现实验。"""

    def __init__(self, num_symbols=16, snr_db=12.0):
        side = int(round(math.sqrt(num_symbols)))
        if side * side != num_symbols:
            raise ValueError("QAM 星座点数量必须是完全平方数。")
        self.num_symbols = num_symbols
        self.snr_db = float(snr_db)
        self.max_level = side - 1
        self.constellation = [(i, j) for i in range(side) for j in range(side)]
        self.symbol_map = {xy: idx for idx, xy in enumerate(self.constellation)}

    def _noise_std(self, device):
        avg_power = (self.num_symbols - 1) / 6.0
        snr = torch.tensor(self.snr_db, device=device)
        return torch.sqrt(avg_power / torch.pow(torch.tensor(10.0, device=device), snr / 10.0) / 2.0)

    def modulate(self, indices):
        coords = torch.empty(indices.numel(), 2, device=indices.device, dtype=torch.float32)
        flat = indices.reshape(-1).long()
        for pos, idx in enumerate(flat.tolist()):
            x, y = self.constellation[int(idx)]
            coords[pos, 0] = float(x)
            coords[pos, 1] = float(y)
        return coords

    def awgn(self, symbols):
        return symbols + self._noise_std(symbols.device) * torch.randn_like(symbols)

    def demodulate(self, symbols):
        rounded = torch.round(symbols).long()
        rounded = rounded.clamp(0, self.max_level)
        decoded = torch.empty(symbols.shape[0], device=symbols.device, dtype=torch.long)
        for pos in range(symbols.shape[0]):
            decoded[pos] = self.symbol_map[(int(rounded[pos, 0]), int(rounded[pos, 1]))]
        return decoded


def power_normalize(z):
    """对离散编码进行简单功率归一化，避免发射功率超过 1。"""

    power = torch.mean(z * z).sqrt()
    if bool(power > 1):
        z = z / power
    return z


class GaussianShapedVectorQuantizer(nn.Module):
    """带高斯目标分布约束的向量量化核心模块。

    输入 latent 形状为 [B, C, H, W]，其中 C 等于 embedding_dim。
    训练阶段会额外计算码字激活分布到高斯目标分布的熵正则化最优传输损失。
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        commitment_cost=0.25,
        epsilon=0.05,
        dual_steps=10,
        dual_lr=0.5,
        hist_temperature=0.5,
        ot_weight=1.0,
        gaussian_mean=None,
        gaussian_std=None,
    ):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.commitment_cost = float(commitment_cost)
        self.epsilon = float(epsilon)
        self.dual_steps = int(dual_steps)
        self.dual_lr = float(dual_lr)
        self.hist_temperature = float(hist_temperature)
        self.ot_weight = float(ot_weight)
        self.gaussian_mean = gaussian_mean
        self.gaussian_std = gaussian_std

        self.codebook = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def gaussian_target(self, device, dtype):
        indices = torch.arange(self.num_embeddings, device=device, dtype=dtype)
        mean = (self.num_embeddings - 1) / 2.0 if self.gaussian_mean is None else self.gaussian_mean
        std = self.num_embeddings / 6.0 if self.gaussian_std is None else self.gaussian_std
        target = torch.exp(-0.5 * ((indices - mean) / std) ** 2)
        return target / target.sum().clamp_min(1e-12)

    def _normalize_prob(self, prob):
        prob = prob.clamp_min(1e-12)
        return prob / prob.sum()

    def _dual_transport_objective(self, phi, src_w, tgt_w, cost):
        src_w = self._normalize_prob(src_w)
        tgt_w = self._normalize_prob(tgt_w)
        log_tgt = torch.log(tgt_w.clamp_min(1e-12)).unsqueeze(0)
        exp_term = (-cost + phi.unsqueeze(0)) / self.epsilon
        logsumexp = torch.logsumexp(log_tgt + exp_term, dim=1)
        return torch.sum(src_w * (-self.epsilon * logsumexp)) + torch.sum(tgt_w * phi)

    def dual_ot_loss(self, src_w, tgt_w, cost):
        """用少量对偶迭代近似码字分布到高斯目标分布的最优传输代价。"""

        src_det = src_w.detach()
        tgt_det = tgt_w.detach()
        cost_det = cost.detach()
        phi = torch.zeros_like(tgt_det, requires_grad=True)

        for _ in range(self.dual_steps):
            obj = self._dual_transport_objective(phi, src_det, tgt_det, cost_det)
            grad_phi = torch.autograd.grad(obj, phi, create_graph=False)[0]
            phi = (phi + self.dual_lr * grad_phi).detach().requires_grad_(True)

        return self._dual_transport_objective(phi.detach(), src_w, tgt_w, cost)

    def _channel_noise(self, encodings, channel):
        if channel is None:
            return torch.zeros_like(encodings)
        clean_indices = torch.argmax(encodings, dim=-1)
        received_indices = channel.demodulate(channel.awgn(channel.modulate(clean_indices))).to(encodings.device)
        return (
            F.one_hot(received_indices, num_classes=self.num_embeddings).float()
            - F.one_hot(clean_indices, num_classes=self.num_embeddings).float()
        )

    def forward(self, inputs, channel=None):
        inputs_bhwc = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs_bhwc.shape
        flat_input = inputs_bhwc.view(-1, self.embedding_dim)

        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self.codebook.weight ** 2, dim=1)
            - 2.0 * torch.matmul(flat_input, self.codebook.weight.t())
        )
        encoding_indices = torch.argmin(distances, dim=1, keepdim=True)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1.0)
        encodings = power_normalize(encodings)

        clean_quantized = torch.matmul(encodings, self.codebook.weight)
        noisy_encodings = encodings + self._channel_noise(encodings, channel)
        noisy_quantized = torch.matmul(noisy_encodings, self.codebook.weight)

        if self.training:
            hard_hist = torch.mean(encodings, dim=0)
            soft_assign = F.softmax(-distances / self.hist_temperature, dim=1)
            soft_hist = torch.mean(soft_assign, dim=0)
            codeword_weight = hard_hist + (soft_hist - soft_hist.detach())
            codeword_weight = self._normalize_prob(codeword_weight)

            target = self.gaussian_target(inputs.device, inputs.dtype)
            support = torch.arange(self.num_embeddings, device=inputs.device, dtype=inputs.dtype).view(-1, 1)
            cost = torch.cdist(support, support, p=2)
            ot_loss = self.ot_weight * self.dual_ot_loss(codeword_weight, target, cost)

            codebook_loss = F.mse_loss(clean_quantized, flat_input.detach())
            commitment_loss = F.mse_loss(clean_quantized.detach(), flat_input)
            loss = codebook_loss + self.commitment_cost * commitment_loss + ot_loss
        else:
            loss = inputs.new_tensor(0.0)

        noisy_quantized = noisy_quantized.view(input_shape)
        quantized = inputs_bhwc + (noisy_quantized - inputs_bhwc).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        return {
            "quantized": quantized,
            "loss": loss,
            "perplexity": perplexity,
            "encodings": encodings,
            "noisy_encodings": noisy_encodings,
            "codeword_hist": avg_probs,
            "target_hist": self.gaussian_target(inputs.device, inputs.dtype),
        }
