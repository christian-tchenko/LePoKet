import argparse
import os
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import resnet

model_names = sorted(name for name in resnet.__dict__
                     if name.startswith("resnet") and callable(resnet.__dict__[name]))

parser = argparse.ArgumentParser(description='Train HKT with ResNet on CIFAR-10')
parser.add_argument('--arch', default='resnet20', choices=model_names)
parser.add_argument('--archparent', default='resnet110', choices=model_names)
parser.add_argument('--epochs', default=64000, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', default=1e-4, type=float)
parser.add_argument('--print-freq', default=50, type=int)
parser.add_argument('--resume', default='save_resnet20/checkpoint_41150.pth', type=str, metavar='PATH')
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
    model = torch.nn.DataParallel(resnet.__dict__[args.arch]()).cuda()

    # Resume logic
    resume_path = args.resume if args.resume else os.path.join(args.save_dir, 'model.pth')
    if os.path.isfile(resume_path):
        print(f"=> Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path)
        args.start_epoch = checkpoint.get('epoch', 0)
        best_prec1 = checkpoint.get('best_prec1', 0)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> No checkpoint found. Training from scratch.")



    parentmodel = torch.nn.DataParallel(resnet.__dict__[args.archparent]()).cuda()
    parent_ckpt = torch.load('pretrained_models/resnet110-1d1ed7c2.th')
    parentmodel.load_state_dict(parent_ckpt['state_dict'])

    cudnn.benchmark = True    
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
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
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transform_test),
        batch_size=100, shuffle=False, num_workers=args.workers, pin_memory=True)

    criterion = nn.CrossEntropyLoss().cuda()
    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)

    print(f"=> Starting training at epoch {args.start_epoch} / {args.epochs}")
    for param_group in optimizer.param_groups:
        if 'initial_lr' not in param_group:
            param_group['initial_lr'] = args.lr

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                milestones=[100, 150], last_epoch=args.start_epoch - 1)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):       
        print(f"=> Epoch {epoch} | LR: {optimizer.param_groups[0]['lr']:.5f}")
        train(train_loader, model, parentmodel, criterion, optimizer, epoch)
        lr_scheduler.step()

        prec1 = validate(val_loader, model, criterion)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint(model, epoch, best_prec1, os.path.join(args.save_dir, f'checkpoint_{epoch}.pth'))

        save_if_best_model(model, epoch, best_prec1, args.save_dir)

def train(loader, model, parent, criterion, optimizer, epoch):
    model.train()
    for i, (x, y) in enumerate(loader):
        x, y = x.cuda(), y.cuda()
        if args.half: x = x.half()

        out, feat = model(x)
        out_p, feat_p = parent(x)
        out_h, feat_h = model(x, out2_h=feat_p, nature='backbone')
        out_ph, _ = parent(x, out2_h=feat_h, nature='head')

        loss = sum(criterion(o, y) for o in [out, out_h, out_ph])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            print(f"Epoch: [{epoch}][{i}/{len(loader)}]\tLoss: {loss.item():.4f}")

def validate(loader, model, criterion):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            if args.half: x = x.half()
            out, _ = model(x)
            pred = out.argmax(dim=1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
    acc = 100. * correct / total
    print(f" * Prec@1 {acc:.3f}")
    return acc

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
        print(f"=> Saving improved model (Prec@1 {best_prec1:.2f}% > {prev:.2f}%)")
        save_checkpoint(model, epoch, best_prec1, model_path)
    else:
        print(f"=> No improvement (Prec@1 {best_prec1:.2f}% <= {prev:.2f}%), model not saved.")

if __name__ == '__main__':
    main()
