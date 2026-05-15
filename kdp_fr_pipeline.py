import torch
import torch.nn.functional as F

from kdp_fr_core import (
    FingerprintFiLMBobDenoiser,
    default_noise_dictionary,
    transmit_indices_with_fingerprint,
)


class KDPFRAlice:
    """KDP-FR 的发射端。

    输入是 VQ 产生的离散码字索引，输出是经过密钥驱动异构信道后的软恢复潜变量、
    硬判决潜变量和信道指纹。公开代码中把密钥抽象为每个样本的 noise_id。
    """

    def __init__(self, noise_dictionary=None):
        self.noise_dictionary = noise_dictionary or default_noise_dictionary()

    def sample_keys(self, batch_size, device):
        return torch.randint(0, len(self.noise_dictionary), (batch_size,), device=device, dtype=torch.long)

    def transmit(self, code_indices, latent_hw, codebook, keys=None):
        if keys is None:
            keys = self.sample_keys(code_indices.shape[0], code_indices.device)
        payload = transmit_indices_with_fingerprint(
            code_indices=code_indices,
            latent_hw=latent_hw,
            codebook=codebook,
            noise_ids=keys,
            noise_dictionary=self.noise_dictionary,
        )
        payload["keys"] = keys
        return payload


class KDPFRBob(torch.nn.Module):
    """KDP-FR 的合法接收端。

    Bob 使用 Alice 提供或共享得到的指纹，对软恢复潜变量进行条件 FiLM 去噪。
    """

    def __init__(self, latent_channels, fingerprint_dim, hidden_channels=128, num_blocks=8):
        super().__init__()
        self.denoiser = FingerprintFiLMBobDenoiser(
            in_channels=latent_channels,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            fingerprint_dim=fingerprint_dim,
        )

    def forward(self, soft_latent, fingerprint):
        return self.denoiser(soft_latent, fingerprint)


class KDPFRSystem(torch.nn.Module):
    """KDP-FR 的最小系统封装，便于在公开代码中展示完整数据流。"""

    def __init__(self, codebook, latent_hw=(4, 4), noise_dictionary=None, hidden_channels=128, num_blocks=8):
        super().__init__()
        self.register_buffer("codebook", codebook.detach().clone())
        self.latent_hw = latent_hw
        self.alice = KDPFRAlice(noise_dictionary=noise_dictionary)

        # 先跑一个空样本推断指纹维度，避免手动计算 fingerprint_dim。
        dummy_indices = torch.zeros(1, latent_hw[0] * latent_hw[1], dtype=torch.long, device=codebook.device)
        dummy_keys = torch.zeros(1, dtype=torch.long, device=codebook.device)
        dummy_payload = self.alice.transmit(dummy_indices, latent_hw, self.codebook, dummy_keys)
        fingerprint_dim = dummy_payload["fingerprint"].shape[1]
        self.bob = KDPFRBob(
            latent_channels=codebook.shape[1],
            fingerprint_dim=fingerprint_dim,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
        )

    def forward(self, code_indices, keys=None):
        payload = self.alice.transmit(code_indices, self.latent_hw, self.codebook, keys)
        recovered_latent = self.bob(payload["soft_latent"], payload["fingerprint"])
        return {
            "recovered_latent": recovered_latent,
            "soft_latent": payload["soft_latent"],
            "hard_latent": payload["hard_latent"],
            "fingerprint": payload["fingerprint"],
            "keys": payload["keys"],
        }


def train_one_kdp_fr_step(system, clean_latent, code_indices, optimizer):
    """一次最小训练步骤：用干净潜变量监督 Bob 的恢复网络。"""

    system.train()
    output = system(code_indices)
    loss = F.mse_loss(output["recovered_latent"], clean_latent)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())
