import torch
import torchvision.datasets as datasets
import torchvision.models as models
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import sys
import os
import shutil
import numpy as np

from settings import get_arguments
from avg import AverageMeter


def load_data(datadir, batch_size, num_workers):
    # -- prepare data ---#
    data_transfrom = {
        'train': transforms.Compose([transforms.RandomSizedCrop(224),
                                     transforms.ToTensor()]),
        'val': transforms.Compose([transforms.CenterCrop(224),
                                   transforms.ToTensor()])}

    dsets = {x: datasets.ImageFolder(os.path.join(datadir, x), transform=data_transfrom[x])
             for x in ['train', 'val']}
    data_loader = {x: data.DataLoader(dsets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers)
                   for x in ['train', 'val']}

    return data_loader

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, is_best, savedir, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'weights/model_best.pth.tar')


def main(model, args):

    #--- load data ---
    data_loader = load_data(args.datadir,args.batch_size, args.num_workers)


    #--- define training settings ---
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss().cuda()



    losses = {x: AverageMeter() for x in ['train', 'val']}
    for epoch in range(1, args.num_epochs+1):

        #---in each epoch, do a train and a validation step---
        for phase in ['train', 'val']:
            if phase =='train':
                model.train()
            else:
                model.eval()
                top1s = AverageMeter()
                best_top1 = 0.0

            losses[phase].reset()
            for i, (image, target) in enumerate(data_loader[phase]):
                images = Variable(image.float().cuda())
                labels = Variable(target.long().cuda())

                outputs = model(images)
                loss = criterion(outputs, labels)
                losses[phase].update(loss.data.cpu().numpy(), args.batch_size)


                if phase== 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    prec1 = accuracy(outputs.data, labels.data)
                    top1s.update(prec1[0].cpu().numpy(), args.batch_size)


        if top1s.avg > best_top1:
            best_top1 = top1s.avg
            is_best = True
        filename = "weights/{0}-{1:02}.pth.tar".format(args.model, epoch)
        save_checkpoint({
            'epoch': epoch,
            'arch': args.model,
            'state_dict': model.state_dict(),
            'best_top1': best_top1,
            'optimizer': optimizer.state_dict(),
        }, is_best, filename)

        print('Epoch:{0}/{1}'
              '\tTrainLoss: {2:.4f}'
              '\tTestLoss: {3:.4f}'
              '\tBestTop1: {4:.4f}'.format(epoch, args.num_epochs + 1, losses['train'].avg, losses['val'].avg, top1s.avg))


if __name__ == '__main__':

    args = get_arguments(sys.argv[1:])

    if args.model == 'alexnet':
        model = models.alexnet(pretrained=True)
        model.classifier._modules['6'] = nn.Linear(4096, 10)
        model = model.cuda()

    elif args.model == 'vgg':
        model = models.vgg19(pretrained=True)
        model.classifier._modules['6'] = nn.Linear(4096, 10)
        model = model.cuda()

    elif args.model == 'resnet':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(2048, 10)
        model = model.cuda()

    elif args.model == 'densenet':
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(1024, 10)
        model = model.cuda()

    main(model, args)

