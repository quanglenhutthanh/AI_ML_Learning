import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import torchvision.models as models

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type=str)
    parser.add_argument('--save_dir',type=str)
    parser.add_argument('--arch', type=str, default='vgg16')
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--hidden_units', type=int)
    return parser.parse_args()

def load_data(args):
    data_dir = args.data_dir
    #data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
        'train' : transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid_test' : transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor,
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_datesets = {
        'train' : datasets.ImageFolder(root=train_dir, transform=data_transforms['train']),
        'valid' : datasets.ImageFolder(root=valid_dir, transform=data_transforms['valid_test']),
        'test' : datasets.ImageFolder(root=test_dir, transform=data_transforms['valid_test']),
    }

    data_loaders = {
        'train' : DataLoader(dataset=image_datesets['train'], batch_size=64, shuffle=True),
        'valid' : DataLoader(dataset=image_datesets['valid'], batch_size=64),
        'test' : DataLoader(dataset=image_datesets['test'], batch_size=64)
    }
    print(image_datesets['train'].class_to_idx)

    return image_datesets, data_loaders

def build_model(args, image_datasets, data_loaders, epochs=5):
    if args.arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif args.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif args.arch == 'resnet':
        model = models.resnet18(pretrained=True)
    elif args.arch == 'mobilenet':
        model = models.mobilenet(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    in_feature_of_pretrained_model = model.classifier[0].in_features
    number_of_data_classes = len(image_datasets['train'].classes)
    classifier = nn.Sequential(
        nn.Linear(in_feature_of_pretrained_model, 2048),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(2048, number_of_data_classes),
        nn.LogSoftmax(dim=1)
    )
    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    for epoch in range(epochs):
        train_loss = 0
        valid_loss = 0
        accurcy = 0
        device = get_device()
        for inputs, labels in data_loaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            

            train_loss += loss.items()




    print('build model')

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def save_checkpoint():
    print('saved')

if __name__ == '__main__':
    args = get_args()
    print('architecture: ' + args.arch)
    image_datasets, data_loaders = load_data(args=args)
    build_model(args, image_datasets=image_datasets, data_loaders=data_loaders)
    # print(f"{args.data_dir} - {args.save_dir} - {args.hidden_units}")





