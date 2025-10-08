import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from resnet import resnet20, resnet110  # from your provided file

# === CIFAR-10 Test Data ===
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# === Evaluation Function ===
def evaluate(model, checkpoint_path, device='cuda'):
    model.to(device)
    model.eval()

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Fix for DataParallel: remove 'module.' prefix from keys
    state_dict = checkpoint['state_dict']
    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)  # model returns (logits, features)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100. * correct / total
    err = 100. - acc
    return acc, err

# === Main Execution ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_20 = resnet20()
acc20, err20 = evaluate(model_20, '/home/christian-tcchenko/Downloads/phd/pytorch_resnet_cifar10/save_resnet20/checkpoint.th', device)

model_110 = resnet110()
acc110, err110 = evaluate(model_110, '/home/christian-tcchenko/Downloads/phd/pytorch_resnet_cifar10/pretrained_models/resnet110-1d1ed7c2.th', device)

print(f"[ResNet-20]  Accuracy: {acc20:.2f}% | Error: {err20:.2f}%")
print(f"[ResNet-110] Accuracy: {acc110:.2f}% | Error: {err110:.2f}%")
