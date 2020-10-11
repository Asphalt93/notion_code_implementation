import os
import time
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from dataset_CIFAR10 import CIFAR10_albumentation

import albumentations as A
from albumentations.pytorch import ToTensor

from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
from model import StackedAutoEncoder, StackedAutoEncoder_noMax

if not os.path.exists('./imgs'):
    os.makedirs('./imgs', exist_ok=True)

def main():
    num_epochs = 400
    batch_size = 32

    a_transform = A.Compose([
        A.GaussNoise(),
        ToTensor(normalize={'mean':[0.485, 0.456, 0.406],
                            'std':[0.229,0.224,0.225]})
    ])

    a_transform_target = A.Compose([
        ToTensor(normalize={'mean': [0.485, 0.456, 0.406],
                            'std': [0.229, 0.224, 0.225]})
    ])


    ### With MaxPooling Layer###

    dataset_train = CIFAR10_albumentation('./data/CIFAR10/', train=True, transform=a_transform, target_transform=a_transform_target, download=True)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)

    model = StackedAutoEncoder_noMax(fine_tune=False).cuda()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    #train_exp3(dataloader_train, num_epochs, model, optimizer, criterion)
    train_exp4(dataloader_train, num_epochs, model, optimizer, criterion)


def train_exp3(dataloader_train, num_epochs, model, optimizer, criterion):

    writer = SummaryWriter('./runs/exp3')

    for epoch in tqdm(range(num_epochs)):

        train_loss = 0
        best_train_loss = 1e9

        model.train()

        for i, data in enumerate(dataloader_train):
            img, target = data
            img = img.cuda()
            target = target.cuda()

            output = model(img)
            loss = criterion(output, target).cuda()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss = train_loss/len(dataloader_train)

        writer.add_scalar('train_loss', train_loss, epoch)
        save_checkpoint(epoch, model, optimizer, 'exp3', f'exp3_epoch_{epoch}')

        if best_train_loss > train_loss:
            best_train_loss = train_loss
            best_epoch = epoch

        print(f"Epoch: {epoch} \t Training Loss: {train_loss:.4f} \t Best Training Loss: {best_train_loss:.4f}({best_epoch})")



def train_exp4(dataloader_train, num_epochs, model, optimizer, criterion):

    writer = SummaryWriter('./runs/exp4')

    for epoch in tqdm(range(num_epochs)):

        train_loss = 0
        best_train_loss = 1e9

        model.train()

        for i, data in enumerate(dataloader_train):
            img, target = data
            img = img.cuda()
            target = target.cuda()

            output = model(img)
            loss = criterion(output, target).cuda()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss = train_loss/len(dataloader_train)

        writer.add_scalar('train_loss', train_loss, epoch)
        save_checkpoint(epoch, model, optimizer, 'exp4', f'exp4_epoch_{epoch}')

        if best_train_loss > train_loss:
            best_train_loss = train_loss
            best_epoch = epoch

        print(f"Epoch: {epoch} \t Training Loss: {train_loss:.4f} \t Best Training Loss: {best_train_loss:.4f}({best_epoch})")


def save_checkpoint(epoch, model, optimizer, exp, filename):
    os.makedirs(f'./save/{exp}', exist_ok=True)
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, os.path.join(f'./save/{exp}/{filename}.ckpt'))

if __name__ == '__main__':
    main()