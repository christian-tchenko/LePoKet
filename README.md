# LePoKet: Learnable Parameter Optimization for Knowledge Transfer

Official PyTorch implementation of **LePoKet**, introduced in:

> **Can Knowledge Transfer Parameters Be Learned? LePoKet for Efficient Robotic Vision**  
> Yanick C. Tchenko, Felix Mohr, Hicham Hadj-Abdelkader, Hedi Tabia  
> arXiv:2609.16637, 2026

📄 **Paper:** https://arxiv.org/abs/2609.16637

LePoKet is a structural knowledge-transfer framework that learns how
knowledge should be inherited from a pretrained parent network by a
compact child network.

Building on **Hereditary Knowledge Transfer (HKT)**, LePoKet introduces
learnable parent-child interactions through a block-wise
**Extract-Transform-Mix (ETM)** interface and a
**Learnable Genetic Attention (LGA)** operator.

Unlike conventional knowledge distillation, LePoKet embeds knowledge
inheritance directly into the forward computation and does not require
auxiliary distillation losses or temperature scaling.

## Results

### Image Classification

| Method | Dataset | Accuracy |
|---|---:|---:|
| ResNet-20 baseline | CIFAR-10 | 91.25% |
| HKT | CIFAR-10 | 92.40% |
| **LePoKet** | **CIFAR-10** | **93.40%** |
| ResNet-20 baseline | CIFAR-100 | 65.30% |
| **LePoKet** | **CIFAR-100** | **74.01%** |

### Optical Flow

Models are trained only on FlyingChairs + FlyingThings3D (C+T).

| Method | Sintel Clean EPE | Sintel Final EPE | KITTI EPE |
|---|---:|---:|---:|
| Compact RAFT | 2.21 | 3.35 | 7.51 |
| 2HKT-RAFT | 1.91 | 3.03 | 7.37 |
| 3HKT-RAFT | 1.90 | 3.08 | 6.45 |
| **LePoKet** | **1.92** | **3.01** | **6.39** |

## Repository Structure

- `learnable_ga.py` - Learnable Genetic Attention
- `resnet_lga.py` - LePoKet/LGA integration for ResNet
- `trainer_lga.py` - LePoKet training pipeline
- `trainer.py` - baseline training
- `trainer_hktcifar100.py` - HKT CIFAR-100 training
- `eval.py` - evaluation
- `pretrained_models/` - pretrained checkpoints
- `run.sh` - training launcher

## Installation

Clone the repository:

    git clone https://github.com/christian-tchenko/LePoKet.git
    cd LePoKet

## HKT

LePoKet builds upon **Hereditary Knowledge Transfer (HKT)**:

https://github.com/christian-tchenko/HKT-ResNet

## Citation

If you find LePoKet useful in your research, please cite:

```bibtex
@article{tchenko2026lepoket,
  title={Can Knowledge Transfer Parameters Be Learned? LePoKet for Efficient Robotic Vision},
  author={Tchenko, Yanick C. and Mohr, Felix and Hadj-Abdelkader, Hicham and Tabia, Hedi},
  journal={arXiv preprint arXiv:2609.16637},
  year={2026}
}
