import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.utils.prune as prune
import argparse
import torch.nn.functional as F
import numpy as np
from PIL import Image
import PIL
import resnet
from models.resnet2 import resnet18 as ResNet18 
from models.mobilenetv2 import mobilenetv2 as MobileNetV2 
from models.shufflenetv2 import shufflenetv2 as ShuffleNetV2 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--epoch', default=200, type=int)
parser.add_argument('--batchsize', default=128, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--magnitude', default=2, type=int)
parser.add_argument('--pre_amount', default=0.2, type=float)
parser.add_argument('--pre_prune', action='store_true')
parser.add_argument('--amount', default=0.2, type=float)
parser.add_argument('--prune', action='store_true')
parser.add_argument('--model_save_path', default='')
parser.add_argument('--model_load_path', default='')
parser.add_argument('--DA', action='store_true')
parser.add_argument('--loading', action='store_true')
parser.add_argument('--m', action='store_true')
parser.add_argument('--s', action='store_true')
args = parser.parse_args()
BATCH_SIZE = 128
LR = 0.1

if args.DA:
    print(args.magnitude)
    transform = transforms.Compose([transforms.RandomCrop(32, padding=4, fill=128),
        transforms.RandomHorizontalFlip(), transforms.RandAugment(2,args.magnitude), transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
else:
    transform = transforms.Compose([transforms.RandomCrop(32, padding=4, fill=128),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='data',
    train=True,
    download=False,
    transform=transform
)
testset = torchvision.datasets.CIFAR10(
    root='data',
    train=False,
    download=False,
    transform=transform_test
)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=args.batchsize,
    shuffle=True,
    num_workers=4
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=args.batchsize,
    shuffle=False,
    num_workers=4
)

net = resnet.resnet20()
# net = ResNet18()
if args.m:
    net = MobileNetV2()
if args.s:
    net = ShuffleNetV2()
# net.load_state_dict(torch.load("/home/lthpc/DA_MC/code/TrainedModel/resnet20_baseline.pth"), strict=True)
if args.pre_prune:
    print("Start to Pre_Prune")
    print(args.pre_amount)
    prune_list = []
    for module in net.modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune_list.append((module, "weight"))

    prune.global_unstructured(
            prune_list,
            pruning_method=prune.L1Unstructured,
            amount=args.pre_amount,
            )

if args.loading:
    ### net.load_state_dict(torch.load("./TrainedModel/MobileNetV2_baseline.pth"), strict=True)
    net.load_state_dict(torch.load(args.model_load_path), strict=True)
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9)
cos_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=5e-6, verbose=True)

if args.prune:
    print("Start to Prune")
    print(args.amount)
    prune_list = []
    for module in net.modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune_list.append((module, "weight"))

    prune.global_unstructured(
            prune_list,
            pruning_method=prune.L1Unstructured,
            amount=args.amount,
            )

acc = 0
with torch.no_grad():
    correct = 0.0
    total = 0.0
    for data in testloader:
        net.eval()
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += float(labels.size(0))
        correct += float((predicted == labels).sum())
    acc = (100 * correct / total)
    print('test accuracy is ', acc)


if __name__ == "__main__":
    best_acc = 0
    epoch = 0
    for epoch in range(args.epoch):
        if epoch in [90, 180]:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10
        net.train()
        sum_loss = 0.0
        sum_c_loss = 0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(trainloader, 0):
            length = len(trainloader)
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += float(labels.size(0))
            correct += float(predicted.eq(labels.data).cpu().sum())
            if i % 50 == 0:
                print('[epoch:%d, iter:%d] Loss: %.03f  | Acc: %.4f%%'
                      % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1),  100 * correct / total))
        print("Waiting Test!")

        acc1 = 0
        with torch.no_grad():
            correct = 0.0
            total = 0.0
            for data in testloader:
                net.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += float(labels.size(0))
                correct += float((predicted == labels).sum())
            acc1 = (100 * correct/total)
            if acc1 > best_acc:
                best_acc = acc1
                # torch.save(net.state_dict(), "./TrainedModel/resnet20_P0_M14.pth")
                # torch.save(net.state_dict(), args.model_save_path)
        print('Test Set Accuracy: %.4f%%' % acc1)
        cos_schedule.step()
    print("Training Finished, TotalEPOCH=%d" % args.epoch)
    print ("Highest Accuracy is ", best_acc)



