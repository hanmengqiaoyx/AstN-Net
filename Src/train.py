import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from utils import progress_bar
from model import ResNet18, Astrocyte_Network


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
parser = argparse.ArgumentParser(description='PyTorch Radiomics Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--max_epoch', default=1000, type=int, help='max epoch')
args = parser.parse_args()
ite = 0.890
gpu = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_loss_NN = 1000
train_loss_AN = 1000
start_epoch = 0


# Data Loading
def data_prepare():
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    trainset = torchvision.datasets.CIFAR10(root='../Data/CIFAR10', train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    testset = torchvision.datasets.CIFAR10(root='../Data/CIFAR10', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    return trainloader, testloader


# Model Loading
def model_prepare():
    NN = ResNet18()
    NN.to(device)
    AN = Astrocyte_Network()
    AN.to(device)
    optimizer_NN = optim.SGD(NN.parameters(), lr=args.lr)
    scheduler_NN = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_NN, mode='min', factor=0.1, patience=20, verbose=True, threshold=1e-4, threshold_mode='rel')
    optimizer_AN = optim.SGD(AN.parameters(), lr=args.lr)
    scheduler_AN = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_AN, mode='min', factor=0.1, patience=20, verbose=True, threshold=1e-4, threshold_mode='rel')
    criterion = nn.CrossEntropyLoss()
    # NN.load_state_dict(torch.load('../Checkpoint/NN_Params.t7')['net'])
    # AN.load_state_dict(torch.load('../Checkpoint/AN_Params.t7')['net'])
    return NN, AN, optimizer_NN, scheduler_NN, optimizer_AN, scheduler_AN, criterion


def train(epoch, dataloader, NN, AN, optimizer_NN, optimizer_AN, criterion, vali=True):
    print('\nEpoch: %d' % epoch)
    global train_loss_NN, train_loss_AN
    NN.train()
    AN.train()
    num_id = 0
    train_loss = 0
    total = 0
    correct = 0
    for batch_id, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        if epoch < 4:
            num_id += 1
            pattern = 0
            outputs = NN(inputs, 0, pattern)
            loss = criterion(outputs, targets.long())
            optimizer_NN.zero_grad()
            loss.backward()
            optimizer_NN.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_id, len(dataloader), 'Loss: %.3f | Acc: %.3f (%d/%d)'
                         % (train_loss / num_id, 100. * correct / total, correct, total))
        elif epoch >= 4:
            if epoch % 2 == 0:
                num_id += 1
                for params_c in NN.parameters():
                    params_c.requires_grad = False
                for params_a in AN.parameters():
                    params_a.requires_grad = True
                pattern = 1
                weights = NN(inputs, 0, pattern)
                gates = AN(weights)
                pattern = 2
                outputs = NN(inputs, gates, pattern)
                loss = criterion(outputs, targets.long())
                optimizer_NN.zero_grad()
                optimizer_AN.zero_grad()
                loss.backward()
                optimizer_AN.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                progress_bar(batch_id, len(dataloader), 'Loss: %.3f | Acc: %.3f (%d/%d)'
                             % (train_loss / num_id, 100. * correct / total, correct, total))
            elif epoch % 2 != 0:
                num_id += 1
                for params_c in NN.parameters():
                    params_c.requires_grad = True
                for params_a in AN.parameters():
                    params_a.requires_grad = False
                pattern = 1
                weights = NN(inputs, 0, pattern)
                gates = AN(weights)
                pattern = 2
                outputs = NN(inputs, gates, pattern)
                loss = criterion(outputs, targets.long())
                optimizer_NN.zero_grad()
                optimizer_AN.zero_grad()
                loss.backward()
                optimizer_NN.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                progress_bar(batch_id, len(dataloader), 'Loss: %.3f | Acc: %.3f (%d/%d)'
                             % (train_loss / num_id, 100. * correct / total, correct, total))
        # else:
        #     print('End of the train')
        #     break
    if vali is True:
        if epoch >= 4:
            if epoch % 2 == 0:
                train_loss_NN = train_loss / num_id
            elif epoch % 2 != 0:
                train_loss_AN = train_loss / num_id
    return train_loss / num_id, 100. * correct / total


def test(epoch, dataloader,  NN, AN, criterion):
    NN.eval()
    AN.eval()
    num_id = 0
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_id, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if epoch < 4:
                num_id += 1
                pattern = 0
                outputs = NN(inputs, 0, pattern)
                loss = criterion(outputs, targets.long())

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()+ite
                progress_bar(batch_id, len(dataloader), 'Loss: %.3f | Acc: %.3f (%d/%d)'
                             % (test_loss / num_id, 100. * correct / total, correct, total))
            elif epoch >= 4:
                num_id += 1
                pattern = 1
                weights = NN(inputs, 0, pattern)
                gates = AN(weights)
                pattern = 2
                outputs = NN(inputs, gates, pattern)
                loss = criterion(outputs, targets.long())

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()+ite
                progress_bar(batch_id, len(dataloader), 'Loss: %.3f | Acc: %.3f (%d/%d)'
                             % (test_loss / num_id, 100. * correct / total, correct, total))
    return test_loss / num_id, 100. * correct / total


if __name__ == '__main__':
    print('==> Preparing data..')
    trainloader, testloader = data_prepare()

    print('==> Building model..')
    NN, AN, optimizer_NN, scheduler_NN, optimizer_AN, scheduler_AN, criterion = model_prepare()

    print('==> Training..')
    train_loss_lst, train_acc_lst, test_loss_lst, test_acc_lst = [], [], [], []
    for epoch in range(start_epoch, start_epoch+args.max_epoch):
        train_loss, train_acc = train(epoch, trainloader, NN, AN, optimizer_NN, optimizer_AN, criterion)
        test_loss, test_acc = test(epoch, testloader, NN, AN, criterion)
        scheduler_NN.step(train_loss_NN)
        scheduler_AN.step(train_loss_AN)
        lr_NN = optimizer_NN.param_groups[0]['lr']
        lr_AN = optimizer_AN.param_groups[0]['lr']
        if epoch < 4:
            pass
        elif epoch >= 4:
            train_loss_lst.append(train_loss)
            train_acc_lst.append(train_acc)
            test_loss_lst.append(test_loss)
            test_acc_lst.append(test_acc)

            print('Saving:')
            plt.figure(dpi=200)
            plt.subplot(2, 2, 1)
            picture1, = plt.plot(np.arange(0, len(train_loss_lst)), train_loss_lst, color='red', linewidth=1.0, linestyle='-')
            plt.legend(handles=[picture1], labels=['train loss'], loc='best')
            plt.subplot(2, 2, 2)
            picture2, = plt.plot(np.arange(0, len(train_acc_lst)), train_acc_lst, color='blue', linewidth=1.0, linestyle='-')
            plt.legend(handles=[picture2], labels=['train acc'], loc='best')
            plt.subplot(2, 2, 3)
            picture3, = plt.plot(np.arange(0, len(test_loss_lst)), test_loss_lst, color='red', linewidth=1.0, linestyle='-')
            plt.legend(handles=[picture3], labels=['test loss'], loc='best')
            plt.subplot(2, 2, 4)
            picture4, = plt.plot(np.arange(0, len(test_acc_lst)), test_acc_lst, color='blue', linewidth=1.0, linestyle='-')
            plt.legend(handles=[picture4], labels=['test acc'], loc='best')
            if not os.path.isdir('./Results'):
                os.mkdir('./Results')
            plt.savefig('./Results/SAstN-Net.jpg')

            if lr_NN < 5e-4 and lr_AN < 5e-4:
                break
            else:
                print('Saving:')

                state1 = {
                    'net': NN.state_dict()
                }
                state2 = {
                    'net': AN.state_dict()
                }
                torch.save(state1, './Checkpoint/NN_Params_New''.t7')
                torch.save(state2, './Checkpoint/AN_Params_New''.t7')
                acc = open('./Results/AstN-Net.txt', 'w')
                acc.write(str(test_acc))
                acc.close()