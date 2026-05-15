import argparse
import json
import os
import pickle
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from gaussian_shaped_vq import SimpleQAMChannel
from vq_models import GaussianVQClassifier


class CIFAR10ClassificationDataset(Dataset):
    """读取 CIFAR-10 python batches，不依赖 torchvision 下载。"""

    def __init__(self, root, train=True, augment=False):
        self.train = train
        self.augment = bool(augment and train)
        batch_names = [f"data_batch_{i}" for i in range(1, 6)] if train else ["test_batch"]
        images, labels = [], []
        for batch_name in batch_names:
            with open(os.path.join(root, batch_name), "rb") as handle:
                batch = pickle.load(handle, encoding="latin1")
            images.append(batch["data"])
            labels.extend(batch["labels"])
        self.images = np.concatenate(images, axis=0).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        self.labels = np.asarray(labels, dtype=np.int64)
        self.mean = np.asarray([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray([0.2023, 0.1994, 0.2010], dtype=np.float32).reshape(3, 1, 1)

    def __len__(self):
        return len(self.labels)

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
        image = (image - self.mean) / self.std
        return torch.from_numpy(image.copy()), torch.tensor(self.labels[index], dtype=torch.long)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gaussian_target(num_embeddings):
    x = np.arange(num_embeddings, dtype=np.float32)
    mean = (num_embeddings - 1) / 2.0
    std = num_embeddings / 6.0
    target = np.exp(-0.5 * ((x - mean) / std) ** 2)
    return target / target.sum()


def js_divergence(p, q):
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


@torch.no_grad()
def evaluate(model, loader, device, args, snr_db):
    model.eval()
    channel = SimpleQAMChannel(args.num_embeddings, snr_db)
    correct, total = 0, 0
    activation_sum = torch.zeros(args.num_embeddings, device=device)
    for step, (images, labels) in enumerate(loader):
        if args.max_eval_batches > 0 and step >= args.max_eval_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
            output = model(images, channel=channel)
        pred = output["logits"].argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += labels.numel()
        activation_sum += output["encodings"].sum(dim=0)
    activation = (activation_sum / activation_sum.sum().clamp_min(1e-12)).detach().cpu().numpy()
    return correct / max(total, 1), activation


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    train_set = CIFAR10ClassificationDataset(args.data_root, train=True, augment=True)
    test_set = CIFAR10ClassificationDataset(args.data_root, train=False, augment=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)

    model = GaussianVQClassifier(
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
        commitment_cost=args.commitment_cost,
        ot_weight=args.ot_weight,
    ).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))
    channel = SimpleQAMChannel(args.num_embeddings, args.train_snr)
    best_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_correct, total_count = 0.0, 0, 0
        for step, (images, labels) in enumerate(train_loader):
            if args.max_train_batches > 0 and step >= args.max_train_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
                output = model(images, channel=channel)
                ce_loss = criterion(output["logits"], labels)
                loss = ce_loss + args.vq_loss_weight * output["vq_loss"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item()) * labels.numel()
            total_correct += int((output["logits"].detach().argmax(dim=1) == labels).sum().item())
            total_count += labels.numel()

        test_acc, activation = evaluate(model, test_loader, device, args, args.train_snr)
        best_acc = max(best_acc, test_acc)
        print(
            f"epoch={epoch} train_loss={total_loss / max(total_count, 1):.4f} "
            f"train_acc={total_correct / max(total_count, 1):.4f} test_acc={test_acc:.4f}"
        )

    target = gaussian_target(args.num_embeddings)
    metrics = {
        "best_acc": best_acc,
        "activation_freq": activation.tolist(),
        "target_gaussian": target.tolist(),
        "js_divergence": js_divergence(activation, target),
    }
    Path(args.output_dir, "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save({"model": model.state_dict()}, Path(args.output_dir, "gaussian_vq_classifier.pth"))

    x = np.arange(args.num_embeddings)
    plt.figure(figsize=(7, 4))
    plt.plot(x, activation, marker="o", label="activation")
    plt.plot(x, target, "k--", label="gaussian target")
    plt.xlabel("Code index")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(args.output_dir, "classification_activation.png"), dpi=180)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/cifar-10-batches-py")
    parser.add_argument("--output-dir", type=str, default="outputs/classification")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-embeddings", type=int, default=16)
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
