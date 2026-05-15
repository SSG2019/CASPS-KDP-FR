import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def noise_std_from_snr(symbols, snr_db):
    """根据符号平均功率和信噪比生成噪声标准差。"""

    power = symbols.pow(2).sum(dim=-1, keepdim=True).mean().clamp_min(1e-12)
    snr = torch.pow(torch.tensor(10.0, device=symbols.device, dtype=symbols.dtype), float(snr_db) / 10.0)
    return torch.sqrt(power / snr / 2.0)


def sample_rician_gain(batch_size, num_tokens, k_factor, min_gain, device, dtype):
    """采样 Rician 衰落增益。"""

    k = float(k_factor)
    los = math.sqrt(k / (k + 1.0))
    scatter = math.sqrt(1.0 / (2.0 * (k + 1.0)))
    real = los + scatter * torch.randn(batch_size, num_tokens, 1, device=device, dtype=dtype)
    imag = scatter * torch.randn(batch_size, num_tokens, 1, device=device, dtype=dtype)
    return torch.sqrt(real.pow(2) + imag.pow(2)).clamp_min(float(min_gain))


def sample_burst_mask(batch_size, num_tokens, burst_length, device, dtype):
    """为每个样本生成一个连续突发遮挡片段。"""

    mask = torch.zeros(batch_size, num_tokens, 1, device=device, dtype=dtype)
    max_start = max(num_tokens - int(burst_length), 0)
    starts = torch.randint(0, max_start + 1, (batch_size,), device=device)
    for row in range(batch_size):
        start = int(starts[row].item())
        end = min(start + int(burst_length), num_tokens)
        mask[row, start:end, 0] = 1.0
    return mask


def sample_quadrant_rotation(batch_size, num_tokens, rotate_prob, device, dtype):
    """随机执行 0、90、180、270 度符号旋转。"""

    rotate_mask = (torch.rand(batch_size, num_tokens, 1, device=device) < float(rotate_prob)).to(dtype)
    turns = torch.randint(0, 4, (batch_size, num_tokens, 1), device=device)
    angles = turns.to(dtype) * (math.pi / 2.0) * rotate_mask
    return torch.cos(angles), torch.sin(angles)


def apply_rotation(symbols, cos_map, sin_map):
    x = symbols[..., :1]
    y = symbols[..., 1:]
    return torch.cat([cos_map * x - sin_map * y, sin_map * x + cos_map * y], dim=-1)


def build_square_qam_constellation(num_embeddings, device, dtype):
    """构造归一化方形 QAM 星座。"""

    side = int(round(math.sqrt(int(num_embeddings))))
    if side * side != int(num_embeddings):
        raise ValueError("码字数量必须是完全平方数，才能映射到方形 QAM。")
    levels = torch.arange(side, device=device, dtype=dtype)
    levels = 2.0 * levels - (side - 1)
    grid_x, grid_y = torch.meshgrid(levels, levels, indexing="ij")
    points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
    avg_power = points.pow(2).sum(dim=1).mean().clamp_min(1e-12)
    return points / torch.sqrt(avg_power)


def indices_to_symbols(code_indices, constellation):
    flat = code_indices.reshape(-1).long()
    symbols = constellation.index_select(0, flat)
    return symbols.view(code_indices.shape[0], code_indices.shape[1], constellation.shape[1])


def hard_demodulate_symbols(symbols, constellation):
    flat_symbols = symbols.reshape(-1, constellation.shape[1])
    distances = torch.cdist(flat_symbols, constellation)
    flat_indices = torch.argmin(distances, dim=1)
    return flat_indices.view(symbols.shape[0], symbols.shape[1])


def recover_latent_from_indices(code_indices, codebook, latent_hw):
    flat = code_indices.reshape(-1).long()
    latent = codebook.index_select(0, flat)
    latent = latent.view(code_indices.shape[0], latent_hw[0], latent_hw[1], codebook.shape[1])
    return latent.permute(0, 3, 1, 2).contiguous()


def simulate_symbol_channel(clean_symbols, entry):
    """执行 KDP-FR 中的异构信道扰动，并返回用于构造指纹的信道状态。"""

    kind = entry["kind"]
    batch_size, num_tokens, _ = clean_symbols.shape
    noise_std = noise_std_from_snr(clean_symbols, float(entry.get("snr_db", 4.0)))
    noise_map = noise_std.expand(batch_size, num_tokens, 1)
    unit_gain = torch.ones(batch_size, num_tokens, 1, device=clean_symbols.device, dtype=clean_symbols.dtype)
    identity_cos = torch.ones_like(unit_gain)
    identity_sin = torch.zeros_like(unit_gain)

    if kind == "awgn":
        observed = clean_symbols + noise_map * torch.randn_like(clean_symbols)
        return observed, unit_gain, noise_map, identity_cos, identity_sin

    if kind == "rician":
        gains = sample_rician_gain(
            batch_size=batch_size,
            num_tokens=num_tokens,
            k_factor=entry.get("rician_k", 6.0),
            min_gain=entry.get("min_gain", 0.2),
            device=clean_symbols.device,
            dtype=clean_symbols.dtype,
        )
        observed = gains * clean_symbols + noise_map * torch.randn_like(clean_symbols)
        return observed, gains, noise_map, identity_cos, identity_sin

    if kind == "impulsive":
        impulse_prob = float(entry.get("impulse_prob", 0.35))
        impulse_scale = float(entry.get("impulse_scale", 5.0))
        impulse_mask = (torch.rand(batch_size, num_tokens, 1, device=clean_symbols.device) < impulse_prob).to(clean_symbols.dtype)
        token_noise = noise_map * (1.0 + impulse_mask * impulse_scale)
        observed = clean_symbols + token_noise * torch.randn_like(clean_symbols)
        return observed, unit_gain, token_noise, identity_cos, identity_sin

    if kind == "burst_erasure":
        burst_mask = sample_burst_mask(
            batch_size=batch_size,
            num_tokens=num_tokens,
            burst_length=entry.get("burst_length", 6),
            device=clean_symbols.device,
            dtype=clean_symbols.dtype,
        )
        erasure_gain = float(entry.get("erasure_gain", 0.02))
        burst_noise_scale = float(entry.get("burst_noise_scale", 2.0))
        gains = unit_gain * (1.0 - burst_mask) + burst_mask * erasure_gain
        token_noise = noise_map * (1.0 + burst_mask * burst_noise_scale)
        observed = gains * clean_symbols + token_noise * torch.randn_like(clean_symbols)
        return observed, gains, token_noise, identity_cos, identity_sin

    if kind == "token_rotation":
        cos_map, sin_map = sample_quadrant_rotation(
            batch_size=batch_size,
            num_tokens=num_tokens,
            rotate_prob=entry.get("rotate_prob", 0.85),
            device=clean_symbols.device,
            dtype=clean_symbols.dtype,
        )
        observed = apply_rotation(clean_symbols, cos_map, sin_map) + noise_map * torch.randn_like(clean_symbols)
        return observed, unit_gain, noise_map, cos_map, sin_map

    raise ValueError("未知信道类型: {}".format(kind))


def build_fingerprint_vectors(noise_ids, gain_map, noise_std, rot_cos, rot_sin, noise_dictionary):
    """把信道类别、噪声强度、增益和旋转状态编码为 Bob 侧恢复指纹。"""

    device = noise_ids.device
    batch_size = noise_ids.shape[0]
    num_noise_types = len(noise_dictionary)
    type_onehot = F.one_hot(noise_ids, num_classes=num_noise_types).float()
    snr_values = torch.tensor(
        [float(noise_dictionary[int(idx)]["snr_db"]) for idx in noise_ids.detach().cpu().tolist()],
        device=device,
        dtype=gain_map.dtype,
    ).unsqueeze(1) / 20.0
    gain_values = gain_map.squeeze(-1)
    noise_values = noise_std.squeeze(-1)
    rot_cos_values = rot_cos.squeeze(-1)
    rot_sin_values = rot_sin.squeeze(-1)
    return torch.cat(
        [
            type_onehot,
            snr_values,
            noise_std.view(batch_size, -1).mean(dim=1, keepdim=True),
            gain_values.mean(dim=1, keepdim=True),
            gain_values.std(dim=1, keepdim=True, unbiased=False),
            gain_values.min(dim=1, keepdim=True).values,
            gain_values.max(dim=1, keepdim=True).values,
            gain_values,
            noise_values,
            rot_cos_values,
            rot_sin_values,
        ],
        dim=1,
    )


def soft_recover_latent(observed_symbols, gain_map, noise_std, rot_cos, rot_sin, constellation, codebook, latent_hw):
    """基于信道指纹进行软解调恢复，保留码字概率而不是只取硬判决。"""

    equalized = observed_symbols / gain_map.clamp_min(1e-4)
    inverse_equalized = torch.cat(
        [
            rot_cos * equalized[..., :1] + rot_sin * equalized[..., 1:],
            -rot_sin * equalized[..., :1] + rot_cos * equalized[..., 1:],
        ],
        dim=-1,
    )
    effective_noise = noise_std / gain_map.clamp_min(1e-4)
    dist2 = torch.sum((inverse_equalized.unsqueeze(2) - constellation.unsqueeze(0).unsqueeze(0)) ** 2, dim=-1)
    temperature = (2.0 * effective_noise.squeeze(-1).pow(2)).clamp_min(1e-4)
    probs = F.softmax(-dist2 / temperature.unsqueeze(-1), dim=-1)
    latent_tokens = torch.matmul(probs, codebook)
    latent = latent_tokens.view(observed_symbols.shape[0], latent_hw[0], latent_hw[1], codebook.shape[1])
    return latent.permute(0, 3, 1, 2).contiguous()


def transmit_indices_with_fingerprint(code_indices, latent_hw, codebook, noise_ids, noise_dictionary):
    """KDP-FR 的核心发射过程：离散调制、异构信道、软恢复和指纹输出。"""

    constellation = build_square_qam_constellation(codebook.shape[0], codebook.device, codebook.dtype)
    clean_symbols = indices_to_symbols(code_indices, constellation)
    batch_size, num_tokens, _ = clean_symbols.shape

    observed_symbols = torch.empty_like(clean_symbols)
    gain_map = torch.empty(batch_size, num_tokens, 1, device=clean_symbols.device, dtype=clean_symbols.dtype)
    noise_std = torch.empty_like(gain_map)
    rot_cos = torch.empty_like(gain_map)
    rot_sin = torch.empty_like(gain_map)

    for noise_id, entry in enumerate(noise_dictionary):
        mask = noise_ids == noise_id
        if not mask.any():
            continue
        observed, gains, noise, cos_map, sin_map = simulate_symbol_channel(clean_symbols[mask], entry)
        observed_symbols[mask] = observed
        gain_map[mask] = gains
        noise_std[mask] = noise
        rot_cos[mask] = cos_map
        rot_sin[mask] = sin_map

    hard_indices = hard_demodulate_symbols(observed_symbols, constellation)
    hard_latent = recover_latent_from_indices(hard_indices, codebook, latent_hw)
    soft_latent = soft_recover_latent(observed_symbols, gain_map, noise_std, rot_cos, rot_sin, constellation, codebook, latent_hw)
    fingerprint = build_fingerprint_vectors(noise_ids, gain_map, noise_std, rot_cos, rot_sin, noise_dictionary)
    return {
        "hard_latent": hard_latent,
        "soft_latent": soft_latent,
        "fingerprint": fingerprint,
        "observed_symbols": observed_symbols,
        "hard_indices": hard_indices,
    }


class FingerprintFiLMBlock(nn.Module):
    """用指纹生成 FiLM 参数，对卷积特征进行条件调制。"""

    def __init__(self, channels, fingerprint_dim, hidden_dim):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.mlp = nn.Sequential(
            nn.Linear(fingerprint_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * channels),
        )

    def forward(self, x, fingerprint):
        y = self.bn(self.conv(x))
        gamma_beta = self.mlp(fingerprint)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return F.relu((1.0 + gamma) * y + beta, inplace=True)


class FingerprintFiLMBobDenoiser(nn.Module):
    """KDP-FR 的 Bob 侧轻量恢复网络。"""

    def __init__(self, in_channels, hidden_channels, num_blocks, fingerprint_dim):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.blocks = nn.ModuleList(
            [
                FingerprintFiLMBlock(
                    channels=hidden_channels,
                    fingerprint_dim=fingerprint_dim,
                    hidden_dim=max(hidden_channels, 128),
                )
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x, fingerprint):
        y = F.relu(self.stem(x), inplace=True)
        for block in self.blocks:
            y = block(y, fingerprint)
        return x - self.head(y)


def default_noise_dictionary(snr_db=4.0):
    """公开示例使用的异构噪声字典。"""

    return [
        {"name": "awgn", "kind": "awgn", "snr_db": float(snr_db)},
        {"name": "rician", "kind": "rician", "snr_db": float(snr_db), "rician_k": 6.0, "min_gain": 0.2},
        {"name": "impulsive", "kind": "impulsive", "snr_db": float(snr_db), "impulse_prob": 0.35, "impulse_scale": 5.0},
        {"name": "burst_erasure", "kind": "burst_erasure", "snr_db": float(snr_db), "burst_length": 6, "erasure_gain": 0.02},
        {"name": "token_rotation", "kind": "token_rotation", "snr_db": float(snr_db), "rotate_prob": 0.85},
    ]
