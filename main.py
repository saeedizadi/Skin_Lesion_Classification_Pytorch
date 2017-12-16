import torch
import torchvision.datasets as datasets
import torchvision.models as models
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import sys
import os
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from visualize import Dashboard

from settings import get_arguments
from avg import AverageMeter


def load_data(datadir, crop_size, batch_size, num_workers):
    # -- prepare data ---#
    data_transfrom = {
        'train': transforms.Compose([transforms.RandomCrop(crop_size),
                                     transforms.RandomHorizontalFlip(),
                                     #transforms.RandomVerticalFlip(),
                                     #transforms.ColorJitter(),
                                     transforms.ToTensor()]),
        'val': transforms.Compose([transforms.CenterCrop(crop_size),
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

def compute_confusion_matrix(model, args):
    data_loader = load_data(args.datadir, args.crop_size, args.batch_size, args.num_workers)
    new_model = nn.Sequential(*list(model.children())[:-1])

    dashboard = Dashboard()

    T = np.empty((0))
    Y = np.empty((0))
    #
    for i, (images, targets) in enumerate(data_loader['train']):
        images = Variable(images.float().cuda(), volatile=True)
        targets = Variable(targets.long().cuda(), volatile=True)
        outputs = model(images)
        _, pred = outputs.topk(1, 1, True, True)
        pred = pred.t()


        T = np.hstack((T, targets.data.cpu().numpy()))
        Y = np.hstack((Y, pred.data.cpu().numpy().squeeze()))

    conf_mat = confusion_matrix(T, Y)
    dashboard.plot_conf_matr(conf_mat)



def extract_features(model, args):
    data_loader = load_data(args.datadir, args.crop_size, args.batch_size, args.num_workers)
    new_model = nn.Sequential(*list(model.children())[:-1])

    dashboard = Dashboard()

    Y = np.empty((0))
    X = np.empty((0, 1024))
#
    for i, (images, targets) in enumerate(data_loader['train']):
        images = Variable(images.float().cuda(), volatile=True)
        targets = Variable(targets.long().cuda(), volatile=True)
        outputs = new_model(images)
        outputs = F.avg_pool2d(outputs, kernel_size=7)
        X = np.vstack((X, outputs.data.cpu().numpy().squeeze()))
        Y = np.hstack((Y, targets.data.cpu().numpy()))

    # X = np.load('train.npy')
    # Y = X[:,0]
    # X = X[:, 1:]
    # X = np.nan_to_num(X)

    #---compute tsne rep. and plot them---
    X, Y = tsne(X, Y)
    dashboard.plot_tsne(X, Y)


def tsne(X, Y):
    X_embedded = TSNE(n_components=2).fit_transform(X,y=Y)
    return X_embedded, Y

def main(model, args, base_parameters=None):

    #--- load data ---
    data_loader = load_data(args.datadir, args.crop_size, args.batch_size, args.num_workers)


    #--- define training settings ---
    if base_parameters is not None:
        optimizer = optim.SGD([
                    {'params': base_parameters},
                    {'params': model.classifier.parameters(), 'lr': args.lr}
                ], lr=args.lr*0.1, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss().cuda()

    losses = {x: AverageMeter() for x in ['train', 'val']}
    best_top1 = 0.0
    best_top5 = 0.0
    for epoch in range(1, args.num_epochs+1):

        #---in each epoch, do a train and a validation step---
        for phase in ['train', 'val']:
            if phase =='train':
                model.train()
            else:
                model.eval()
                top1s = AverageMeter()
                top5s = AverageMeter()

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
                    prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1,5))
                    top1s.update(prec1[0], args.batch_size)
                    top5s.update(prec5[0], args.batch_size)

        if top1s.avg > best_top1:
            best_top1 = top1s.avg
            best_top5 = top5s.avg

            filename = "weights/{0}-best.pth.tar".format(args.model)
            state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_top1': best_top1,
                    'optimizer': optimizer.state_dict(),
                    }
            torch.save(state, filename)

        if epoch % args.log_step == 0:
            filename = "weights/{0}-{1:02}.pth.tar".format(args.model, epoch)
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_top1': best_top1,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, filename)

        print('Epoch:{0}/{1}'
              '\tTrainLoss: {2:.4f}'
              '\tTestLoss: {3:.4f}'
              '\tBestTop1: {4:.4f}'
              '\tBestTop5: {5:.4f}'.format(epoch, args.num_epochs, losses['train'].avg, losses['val'].avg, best_top1, best_top5))


if __name__ == '__main__':

    args = get_arguments(sys.argv[1:])

    if args.model == 'alexnet':
        model = models.alexnet(pretrained=True)
        model.classifier._modules['6'] = nn.Linear(4096, 10)
        ignored_params = list(map(id, model.classifier.parameters()))
	base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

        model = model.cuda()

    elif args.model == 'vgg':
        model = models.vgg19(pretrained=True)
        model.classifier._modules['6'] = nn.Linear(4096, 10)
        #---we will use larger lr for fully connected layers---
        ignored_params = list(map(id, model.classifier.parameters()))
	base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

        model = model.cuda()

    elif args.model == 'resnet':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(2048, 10)
        ignored_params = list(map(id, model.fc.parameters()))
	base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        model = model.cuda()

    elif args.model == 'densenet':
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(1024, 10)
        ignored_params = list(map(id, model.classifier.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

        checkpoint = torch.load('weights/bests/densenet-best.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
        model = model.cuda()



    #---Parallel training on several GPUs---
    #model = nn.DataParallel(model, device_ids=[0,1]).cuda()
    #main(model, args, base_params)
    # extract_features(model, args)
    compute_confusion_matrix(model, args)

