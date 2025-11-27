import argparse
import os
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import resnet  # your modified resnet.py with num_classes support

# Collect available architectures dynamically
model_names = sorted(
    name for name in resnet.__dict__
    if name.startswith("resnet") and callable(resnet.__dict__[name])
)

parser = argparse.ArgumentParser(description='Train PURE HKT with ResNet on CIFAR-100')
parser.add_argument('--arch', default='resnet20', choices=model_names)
parser.add_argument('--archparent', default='resnet110', choices=model_names)
parser.add_argument('--epochs', default=64000, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', default=1e-4, type=float)
parser.add_argument('--print-freq', default=50, type=int)
parser.add_argument('--resume', default='', type=str, metavar='PATH')
parser.add_argument('--save-dir', default='save_hkt_cifar100', type=str)
parser.add_argument('--save-every', default=10, type=int)
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--half', action='store_true')
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--start-epoch', default=0, type=int)
args = parser.parse_args()

best_prec1 = 0


##########################################
# Main training function
##########################################
def main():
    global best_prec1

    os.makedirs(args.save_dir, exist_ok=True)

    ##########################################
    # Child student model (ResNet20)
    ##########################################
    print(f"=> Creating CHILD model '{args.arch}' for CIFAR-100")
    model = torch.nn.DataParallel(
        resnet.__dict__[args.arch](num_classes=100)
    ).cuda()

    ##########################################
    # Resume if checkpoint exists
    ##########################################
    resume_path = args.resume if args.resume else os.path.join(args.save_dir, 'model.pth')
    if os.path.isfile(resume_path):
        print(f"=> Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path)
        args.start_epoch = checkpoint.get('epoch', 0)
        best_prec1 = checkpoint.get('best_prec1', 0)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> No checkpoint found. Training from scratch.")

    ##########################################
    # Parent teacher model (ResNet110)
    ##########################################
    print(f"=> Loading PARENT model '{args.archparent}'")
    parentmodel = torch.nn.DataParallel(
        resnet.__dict__[args.archparent](num_classes=100)
    ).cuda()

    # You must provide a CIFAR-100-compatible parent checkpoint.
    parent_ckpt_path = 'pretrained_models/resnet110_cifar100.pth'
    if os.path.isfile(parent_ckpt_path):
        parent_ckpt = torch.load(parent_ckpt_path)
        parentmodel.load_state_dict(parent_ckpt['state_dict'])
        print("=> Loaded CIFAR-100 parent checkpoint.")
    else:
        print("=> WARNING: Parent CIFAR-100 checkpoint NOT FOUND.")
        print(f"   Expected at: {parent_ckpt_path}")
        print("   HKT will still run but performance will be greatly reduced!")

    cudnn.benchmark = True

    ##########################################
    # CIFAR-100 normalization
    ##########################################
    normalize = transforms.Normalize(
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761)
    )

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    ##########################################
    # Dataloaders
    ##########################################
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data', train=True, download=True, transform=transform_train),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data', train=False, transform=transform_test),
        batch_size=100, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    ##########################################
    # Loss & Optimizer
    ##########################################
    criterion = nn.CrossEntropyLoss().cuda()
    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )


    # 🔧 IMPORTANT: set initial_lr for resuming
    for pg in optimizer.param_groups:
        if 'initial_lr' not in pg:
            pg['initial_lr'] = args.lr  

    ##########################################
    # LR Scheduler (scaled for 64k epochs)
    ##########################################
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[20000, 40000], last_epoch=args.start_epoch - 1
    )

    print(f"=> Starting training at epoch {args.start_epoch} / {args.epochs}")

    ##########################################
    # Evaluation mode only
    ##########################################
    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    ##########################################
    # Training loop
    ##########################################
    for epoch in range(args.start_epoch, args.epochs):
        print(f"=> Epoch {epoch} | LR: {optimizer.param_groups[0]['lr']:.5f}")

        train(train_loader, model, parentmodel, criterion, optimizer, epoch)
        lr_scheduler.step()

        prec1 = validate(val_loader, model, criterion)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        # Save periodic checkpoints
        if epoch > 0 and epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.save_dir, f'CIFAR100_checkpoint_{epoch}.pth')
            save_checkpoint(model, epoch, best_prec1, ckpt_path)

        # Save best model
        save_if_best_model(model, epoch, best_prec1, args.save_dir)


##########################################
# HKT Training Step
##########################################
def train(loader, model, parent, criterion, optimizer, epoch):
    model.train()
    parent.eval()

    for i, (x, y) in enumerate(loader):
        x, y = x.cuda(), y.cuda()
        if args.half:
            x = x.half()

        ##########################################
        # HKT Forward logic (pure, no LGA)
        ##########################################
        out, feat = model(x)
        out_p, feat_p = parent(x)

        # Feature guidance (backbone transfer)
        out_h, feat_h = model(x, out2_h=feat_p, nature='backbone')

        # Parent head updated with child features (head transfer)
        out_ph, _ = parent(x, out2_h=feat_h, nature='head')

        ##########################################
        # TOTAL HKT LOSS
        ##########################################
        loss = sum(criterion(o, y) for o in [out, out_h, out_ph])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            print(f"Epoch: [{epoch}][{i}/{len(loader)}]\tLoss: {loss.item():.4f}")


##########################################
# Validation
##########################################
def validate(loader, model, criterion):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            out, _ = model(x)
            pred = out.argmax(dim=1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)

    acc = 100.0 * correct / total
    print(f" * Prec@1 {acc:.3f}")
    return acc


##########################################
# Checkpointing
##########################################
def save_checkpoint(model, epoch, best_prec1, path):
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1
    }, path)


def save_if_best_model(model, epoch, best_prec1, save_dir):
    model_path = os.path.join(save_dir, 'model.pth')
    prev = 0

    if os.path.isfile(model_path):
        prev_ckpt = torch.load(model_path, map_location='cpu')
        prev = prev_ckpt.get('best_prec1', 0)

    if best_prec1 > prev:
        print(f"=> Saving improved BEST model (Prec@1 {best_prec1:.2f}% > {prev:.2f}%)")
        save_checkpoint(model, epoch, best_prec1, model_path)
    else:
        print(f"=> No improvement (Prec@1 {best_prec1:.2f}% <= {prev:.2f}%), not saved.")


##########################################
# Main entry point
##########################################
if __name__ == '__main__':
    main()

