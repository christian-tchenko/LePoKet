import argparse
import os
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pandas as pd  # Excel logging

import resnet
import resnet_lga  # updated import

# Get available model names from resnet_lga
model_names = sorted(name for name in resnet_lga.__dict__
                     if name.startswith("resnet") and callable(resnet_lga.__dict__[name]))

parser = argparse.ArgumentParser(description='Train HKT + Genetic Attention ResNet on CIFAR-10')
parser.add_argument('--arch', default='resnet20', choices=model_names)
parser.add_argument('--archparent', default='resnet110', choices=model_names)
parser.add_argument('--epochs', default=64000, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', default=1e-4, type=float)
parser.add_argument('--print-freq', default=50, type=int)
parser.add_argument('--resume', default='', type=str, metavar='PATH')
parser.add_argument('--save-dir', default='save_temp', type=str)
parser.add_argument('--save-every', default=10, type=int)
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--half', action='store_true')
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--start-epoch', default=0, type=int)
args = parser.parse_args()

best_prec1 = 0

def main():
    global best_prec1
    os.makedirs(args.save_dir, exist_ok=True)

    excel_path = os.path.join(args.save_dir, "training_log.xlsx")

    print(f"=> Creating model '{args.arch}' with Learnable Genetic Attention")
    # updated to use resnet_lga
    model = torch.nn.DataParallel(resnet_lga.__dict__[args.arch](use_lga=True)).cuda()

    resume_path = args.resume or os.path.join(args.save_dir, 'model.pth')
    if os.path.isfile(resume_path):
        print(f"=> Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path)
        args.start_epoch = checkpoint.get('epoch', 0)
        best_prec1 = checkpoint.get('best_prec1', 0)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> No checkpoint found. Training from scratch.")

    print(f"=> Loading parent model '{args.archparent}'")
    parentmodel = torch.nn.DataParallel(resnet.__dict__[args.archparent]()).cuda()
    parent_ckpt = torch.load('pretrained_models/resnet110-1d1ed7c2.th')
    parentmodel.load_state_dict(parent_ckpt['state_dict'])

    cudnn.benchmark = True

    normalize = transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True, transform=transform_train),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transform_test),
        batch_size=100, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    criterion = nn.CrossEntropyLoss().cuda()
    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 150], last_epoch=args.start_epoch - 1)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    print(f"=> Starting training at epoch {args.start_epoch} / {args.epochs}")
    for param_group in optimizer.param_groups:
        if 'initial_lr' not in param_group:
            param_group['initial_lr'] = args.lr

    for epoch in range(args.start_epoch, args.epochs):
        print(f"=> Epoch {epoch} | LR: {optimizer.param_groups[0]['lr']:.5f}")

        train_loss, lam_val = train(train_loader, model, parentmodel, criterion, optimizer, epoch)
        lr_scheduler.step()
        prec1 = validate(val_loader, model, criterion)

        best_prec1 = max(prec1, best_prec1)

        # Log results to Excel
        log_to_excel(excel_path, epoch, prec1, train_loss, lam_val)

        if epoch % args.save_every == 0:
            save_checkpoint(model, epoch, best_prec1,
                            os.path.join(args.save_dir, f'checkpoint_{epoch}.pth'))
        save_if_best_model(model, epoch, best_prec1, args.save_dir)


if __name__ == '__main__':
    main()
