import argparse
import json
import math
import os
import pickle
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from gaussian_shaped_vq import SimpleQAMChannel
from vq_models import GaussianVQReconstructor


class CIFAR10ReconstructionDataset(Dataset):
    """CIFAR-10 重建任务数据集，标签不参与训练。"""

    def __init__(self, root, train=True, augment=False):
        self.augment = bool(augment and train)
        batch_names = [f"data_batch_{i}" for i in range(1, 6)] if train else ["test_batch"]
        images = []
        for batch_name in batch_names:
            with open(os.path.join(root, batch_name), "rb") as handle:
                batch = pickle.load(handle, encoding="latin1")
            images.append(batch["data"])
        self.images = np.concatenate(images, axis=0).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0

    def __len__(self):
        return len(self.images)

    def _augment(self, image):
        padded = np.pad(image, ((0, 0), (4, 4), (4, 4)), mode="reflect")
        top = np.random.randint(0, 9)
        left = np.random.randint(0, 9)
        image = padded[:, top : top + 32, left : left + 32]
        if np.random.rand() < 0.5:
            image = image[:, :, ::-1].copy()
        return image

    def __getitem__(self, index):
        image = self.images[index]
        if self.augment:
            image = self._augment(image)
        return torch.from_numpy(image.copy())


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mse_to_psnr(mse):
    return 10.0 * math.log10(1.0 / max(float(mse), 1e-12))


def gaussian_target(num_embeddings):
    x = np.arange(num_embeddings, dtype=np.float32)
    mean = (num_embeddings - 1) / 2.0
    std = num_embeddings / 6.0
    target = np.exp(-0.5 * ((x - mean) / std) ** 2)
    return target / target.sum()


@torch.no_grad()
def evaluate(model, loader, device, args, snr_db):
    model.eval()
    channel = SimpleQAMChannel(args.num_embeddings, snr_db)
    total_mse, total_pixels = 0.0, 0
    activation_sum = torch.zeros(args.num_embeddings, device=device)
    for step, images in enumerate(loader):
        if args.max_eval_batches > 0 and step >= args.max_eval_batches:
            break
        images = images.to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
            output = model(images, channel=channel)
        total_mse += float(F.mse_loss(output["recon"], images, reduction="sum").item())
        total_pixels += images.numel()
        activation_sum += output["encodings"].sum(dim=0)
    mse = total_mse / max(total_pixels, 1)
    activation = (activation_sum / activation_sum.sum().clamp_min(1e-12)).detach().cpu().numpy()
    return mse, mse_to_psnr(mse), activation


@torch.no_grad()
def save_reconstruction_grid(model, loader, device, args, output_path):
    model.eval()
    channel = SimpleQAMChannel(args.num_embeddings, args.train_snr)
    images = next(iter(loader))[:8].to(device)
    if device.type == "cuda":
        images = images.contiguous(memory_format=torch.channels_last)
    recon = model(images, channel=channel)["recon"].clamp(0.0, 1.0)
    originals = images.detach().cpu().numpy()
    recon = recon.detach().cpu().numpy()
    row = []
    for idx in range(originals.shape[0]):
        row.append(np.transpose(originals[idx], (1, 2, 0)))
        row.append(np.transpose(recon[idx], (1, 2, 0)))
    grid = np.concatenate(row, axis=1)
    plt.figure(figsize=(14, 3))
    plt.imshow(np.clip(grid, 0.0, 1.0))
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0)
    plt.close()


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    train_set = CIFAR10ReconstructionDataset(args.data_root, train=True, augment=True)
    test_set = CIFAR10ReconstructionDataset(args.data_root, train=False, augment=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)

    model = GaussianVQReconstructor(
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
        commitment_cost=args.commitment_cost,
        ot_weight=args.ot_weight,
    ).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))
    channel = SimpleQAMChannel(args.num_embeddings, args.train_snr)
    best_psnr = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_count = 0.0, 0
        for step, images in enumerate(train_loader):
            if args.max_train_batches > 0 and step >= args.max_train_batches:
                break
            images = images.to(device, non_blocking=True)
            if device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
                output = model(images, channel=channel)
                recon_loss = F.mse_loss(output["recon"], images)
                loss = recon_loss + args.vq_loss_weight * output["vq_loss"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * images.size(0)
            total_count += images.size(0)

        mse, psnr, activation = evaluate(model, test_loader, device, args, args.train_snr)
        best_psnr = max(best_psnr, psnr)
        print(f"epoch={epoch} train_loss={total_loss / max(total_count, 1):.5f} mse={mse:.6f} psnr={psnr:.3f}")

    target = gaussian_target(args.num_embeddings)
    metrics = {
        "best_psnr": best_psnr,
        "activation_freq": activation.tolist(),
        "target_gaussian": target.tolist(),
    }
    Path(args.output_dir, "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save({"model": model.state_dict()}, Path(args.output_dir, "gaussian_vq_reconstructor.pth"))
    save_reconstruction_grid(model, test_loader, device, args, Path(args.output_dir, "reconstruction_examples.png"))

    x = np.arange(args.num_embeddings)
    plt.figure(figsize=(7, 4))
    plt.plot(x, activation, marker="o", label="activation")
    plt.plot(x, target, "k--", label="gaussian target")
    plt.xlabel("Code index")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(args.output_dir, "reconstruction_activation.png"), dpi=180)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/cifar-10-batches-py")
    parser.add_argument("--output-dir", type=str, default="outputs/reconstruction")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--num-embeddings", type=int, default=64)
    parser.add_argument("--commitment-cost", type=float, default=0.25)
    parser.add_argument("--ot-weight", type=float, default=1.0)
    parser.add_argument("--train-snr", type=float, default=12.0)
    parser.add_argument("--vq-loss-weight", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=-1)
    parser.add_argument("--max-eval-batches", type=int, default=-1)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
