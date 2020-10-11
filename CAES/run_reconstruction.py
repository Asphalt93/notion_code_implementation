import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from dataset__clsCIFAR10 import CIFAR10_albumentation
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensor

from tqdm import tqdm
from model import StackedAutoEncoder, StackedAutoEncoder_noMax

if not os.path.exists('./imgs'):
    os.makedirs('./imgs', exist_ok=True)

def to_img(x):
    x = x.view(x.size(0), 3, 32, 32)
    return x

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

    dataset_valid = CIFAR10_albumentation('./data/CIFAR10/', train=False, transform=a_transform, target_transform=a_transform_target, download=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)


    model = StackedAutoEncoder(fine_tune=True).cuda()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    ckpt_path = './save/exp4/exp4_epoch_10.ckpt'
    checkpoint = torch.load(ckpt_path)

    model_parameter = model.state_dict()
    # pretrained_param = {k: v for k, v in checkpoint['state_dict'].items() if k in model_parameter}
    # model_parameter.update(pretrained_param)
    model.load_state_dict(model_parameter)
    for param in model.encoder.parameters():
        param.requires_grad = False

    # optimizer.load_state_dict(checkpoint['optimizer'])

    train_exp1(dataloader_train, num_epochs, model, optimizer, criterion, dataloader_valid)

def train_exp1(dataloader_train, num_epochs, model, optimizer, criterion, dataloader_valid):

    writer = SummaryWriter('./runs/exp5_val')
    for epoch in tqdm(range(num_epochs)):

        train_loss = 0
        best_train_loss = 1e9

        model.train()

        for i, data in enumerate(dataloader_train):
            img, target = data
            img = img.cuda()
            target = torch.nn.functional.one_hot(target.cuda(),num_classes=10)
            target = target.type(torch.float32)
            output = model(img)
            loss = criterion(output, target).cuda()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss = train_loss/len(dataloader_train)

        writer.add_scalar('train_loss', train_loss, epoch)
        save_checkpoint(epoch, model, optimizer, 'exp5', f'exp5_epoch_{epoch}')

        if best_train_loss > train_loss:
            best_train_loss = train_loss
            best_epoch = epoch

        valid_loss, valid_acc = valid_exp(dataloader_valid, model, criterion, writer)

        print(
            f"Epoch: {epoch} \t Training Loss: {train_loss:.4f} \t Best Training Loss: {best_train_loss:.4f}({best_epoch}) \t Valid_loss: {valid_loss} \t Valid_acc: {valid_acc}")

def valid_exp(dataloader_valid, model, criterion, writer):
    valid_acc = 0
    valid_loss = 0
    best_valid_loss = 1e9
    with torch.no_grad():
        model.eval()

        for i, data in enumerate(dataloader_valid):
            img, target = data
            img = img.cuda()
            target = target.cuda()


            output = model(img)

            label_output = output.argmax(dim=1)
            acc_output = (label_output == target)
            acc_now = acc_output.sum().float() / float(target.size(0))
            valid_acc += acc_now
            target = torch.nn.functional.one_hot(target.cuda(),num_classes=10)
            target = target.type(torch.float32)

            loss = criterion(output, target).cuda()
            valid_loss += loss.item()

    valid_loss = valid_loss/len(dataloader_valid)
    valid_acc = valid_acc/len(dataloader_valid)
    writer.add_scalar('valid_loss', valid_loss, i)
    writer.add_scalar('valid_acc', valid_acc, i)

    if best_valid_loss > valid_loss:
        best_valid_loss = valid_loss

    return valid_loss, valid_acc





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





