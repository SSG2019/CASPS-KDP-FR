# Gaussian-Shaped VQ and KDP-FR for Privacy-Preserving Semantic Communication

This repository provides the public implementation of the core algorithms proposed in our paper. It includes only our method, and minimal training scripts.

## System Overview

Overall framework of the proposed privacy-aware digital ToSC system. Alice, Bob, and Eve denote the semantic transmitter, the legitimate receiver,and the passive eavesdropper, respectively. Alice maps the source sample x to discrete semantic indices through the semantic encoder and VQ codebook,while CASPS shapes their activation distribution toward a Gaussian target before modulation. KDP-FR then protects the transmitted semantic symbols through key-conditioned perturbation; Bob performs fingerprint-assisted recovery, whereas Eve conducts fingerprint-blind inference from the intercepted signal.

![System model](assets/main.svg)

## Released Files

| File | Description |
| --- | --- |
| `gaussian_shaped_vq.py` | Gaussian-shaped vector quantization module with an entropy-regularized dual optimal transport loss. |
| `vq_models.py` | Classification and reconstruction models using the proposed VQ module. |
| `train_classification.py` | Training script for the CIFAR-10 classification task. |
| `train_reconstruction.py` | Training script for the CIFAR-10 reconstruction task. |
| `kdp_fr_core.py` | Low-level KDP-FR implementation, including heterogeneous channels, fingerprint construction, soft recovery, and the FiLM denoiser. |
| `kdp_fr_pipeline.py` | A readable high-level KDP-FR interface organized as Alice, Bob, and a one-step training function. |

## Environment

The code was tested with Python 3.7 and PyTorch 1.13. Newer PyTorch versions should also work.

```bash
pip install -r requirements.txt
```

## Dataset

The training scripts use the original CIFAR-10 Python batch format:

```text
data/cifar-10-batches-py/
  data_batch_1
  data_batch_2
  data_batch_3
  data_batch_4
  data_batch_5
  test_batch
```

You can also provide a local absolute path with `--data-root`.

## Classification Training

```bash
python train_classification.py \
  --data-root data/cifar-10-batches-py \
  --output-dir outputs/classification \
  --epochs 30 \
  --batch-size 512
```

For a quick smoke test:

```bash
python train_classification.py \
  --data-root data/cifar-10-batches-py \
  --output-dir outputs/classification_smoke \
  --epochs 1 \
  --max-train-batches 1 \
  --max-eval-batches 1
```

## Reconstruction Training

```bash
python train_reconstruction.py \
  --data-root data/cifar-10-batches-py \
  --output-dir outputs/reconstruction \
  --epochs 30 \
  --batch-size 256
```

For a quick smoke test:

```bash
python train_reconstruction.py \
  --data-root data/cifar-10-batches-py \
  --output-dir outputs/reconstruction_smoke \
  --epochs 1 \
  --max-train-batches 1 \
  --max-eval-batches 1
```

## KDP-FR 

The recommended entry point is `kdp_fr_pipeline.py`. It separates KDP-FR into three readable components:

- `KDPFRAlice`: selects the key-driven heterogeneous perturbation and generates the channel fingerprint.
- `KDPFRBob`: uses the fingerprint for FiLM-conditioned latent restoration.
- `KDPFRSystem`: connects Alice-side transmission and Bob-side recovery for a compact end-to-end example.

Example:

```python
import torch
from kdp_fr_pipeline import KDPFRSystem, train_one_kdp_fr_step

codebook = torch.randn(16, 128)
code_indices = torch.randint(0, 16, (8, 16))
clean_latent = codebook[code_indices.reshape(-1)].view(8, 4, 4, 128).permute(0, 3, 1, 2)

system = KDPFRSystem(codebook=codebook, latent_hw=(4, 4), hidden_channels=32, num_blocks=2)
optimizer = torch.optim.AdamW(system.bob.parameters(), lr=1e-3)
loss = train_one_kdp_fr_step(system, clean_latent, code_indices, optimizer)
```

In this abstraction, `code_indices` are the discrete codeword indices produced by the VQ module, and `codebook` is the learned VQ codebook. Alice maps codewords to modulation symbols and applies key-driven heterogeneous perturbations. Bob then uses the fingerprint to restore the latent representation.

## Scope of This Release

The goal is to make the two proposed algorithmic components transparent and runnable:

1. Gaussian-shaped VQ for codeword distribution shaping.
2. KDP-FR for key-driven privacy protection and fingerprint-guided Bob-side recovery.
