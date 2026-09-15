# LePoKet: Learnable Parameter Optimization for Knowledge Transfer

Official PyTorch implementation of **LePoKet (Learnable Parameter Optimization
for Knowledge Transfer)**, a framework for learning how knowledge is transferred
from a pretrained parent network to a compact child network.

LePoKet builds on **Hereditary Knowledge Transfer (HKT)** by replacing fixed
parent-child interactions with learnable transfer parameters. Knowledge
inheritance is integrated directly into the forward computation through a
block-wise **Extract-Transform-Mix (ETM)** interface and a
**Learnable Genetic Attention (LGA)** operator.

The framework is evaluated on image classification using CIFAR-10/CIFAR-100
with ResNet parent-child architectures, and on dense motion estimation using
compact RAFT-based optical flow.

## Main Results

### Image Classification

| Method | Dataset | Accuracy |
|--------|---------|----------|
| ResNet-20 baseline | CIFAR-10 | 91.25% |
| HKT | CIFAR-10 | 92.40% |
| **LePoKet** | **CIFAR-10** | **93.40%** |
| ResNet-20 baseline | CIFAR-100 | 65.30% |
| **LePoKet** | **CIFAR-100** | **74.01%** |

### Optical Flow

Models are trained only on FlyingChairs + FlyingThings3D (C+T).

| Method | Sintel Clean EPE | Sintel Final EPE | KITTI EPE |
|--------|-----------------:|-----------------:|----------:|
| Compact RAFT baseline | 2.21 | 3.35 | 7.51 |
| 2HKT-RAFT | 1.91 | 3.03 | 7.37 |
| 3HKT-RAFT | 1.90 | 3.08 | 6.45 |
| **LePoKet** | **1.92** | **3.01** | **6.39** |

## Repository Structure

The main components include:

- `learnable_ga.py` - Learnable Genetic Attention module
- `resnet_lga.py` - ResNet integration of LePoKet/LGA
- `trainer_lga.py` - training pipeline for LePoKet
- `trainer.py` - standard baseline training
- `trainer_hktcifar100.py` - HKT-based CIFAR-100 training
- `eval.py` - model evaluation
- `pretrained_models/` - pretrained model checkpoints
- `run.sh` - training launcher

## Running the Code

Clone the repository:

    git clone https://github.com/christian-tchenko/LePoKet.git
    cd LePoKet

Then run:

    chmod +x run.sh
    ./run.sh

## Method

Given a frozen parent network and a compact child network, LePoKet learns
the parameters controlling parent-child knowledge transfer jointly with
the child model.

Unlike conventional knowledge distillation, LePoKet does not require
teacher-logit matching, auxiliary feature-matching losses, or temperature
scaling. The transfer mechanism is optimized directly from the downstream
task objective.

## HKT

LePoKet extends **Hereditary Knowledge Transfer (HKT)**.

HKT implementation:
https://github.com/christian-tchenko/HKT-ResNet

## Paper

**Can Knowledge Transfer Parameters Be Learned? LePoKet for Efficient Robotic Vision**

Yanick C. Tchenko, Felix Mohr, Hicham H. Abdelkader, Hedi Tabia

arXiv link: coming soon.

## Citation

The BibTeX citation will be added when the arXiv identifier is available.

## License

See the `LICENSE` file for details.
