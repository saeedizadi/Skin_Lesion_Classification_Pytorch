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

            losses[phase].reset()
            for i, (image, target) in enumerate(data_loader[phase]):
                images = Variable(image.float().cuda())
                labels = Variable(target.long().cuda())

                output = model(images)
                loss = criterion(output, labels)
                losses[phase].update(loss.data.cpu().numpy(), args.batch_size)

                if phase== 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        print('Epoch:{0}/{1}'
              '\tTrainLoss: {2:.4f}'
              '\tTestLoss: {3:.4f}'.format(epoch, args.num_epochs + 1, losses['train'].avg,losses['val'].avg))


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

