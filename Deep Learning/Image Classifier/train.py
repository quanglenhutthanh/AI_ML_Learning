import argparse
from tqdm import tqdm
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

def validate_model(model, validate_data):
    device = get_device()
    valid_loss = 0
    correct = 0
    total = 0
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    model.eval()
    with torch.no_grad():
        for inputs, labels in validate_data:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(input)
            loss = criterion(outputs, labels)
            valid_loss += loss.item() * input.size[0]

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    valid_loss = valid_loss/total
    accuracy = correct/total

    return valid_loss, accuracy

def train_model(args, image_datasets, data_loaders, epochs=5):
    device = get_device()
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
    model.to(device=device)
    for epoch in range(epochs):
        train_loss = 0
        valid_loss = 0
        accuracy = 0
        progress_bar = tqdm(enumerate(data_loaders['train']), total=len(data_loaders['train']), desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

            progress_bar.set_postfix({'batch_loss': loss.item()})
        valid_loss, accuracy = validate_model(model=model, validate_data=data_loaders['valid'])

    print(f"Epoch {epoch+1}/{epochs}.. "
        f"Train loss: {train_loss/len(data_loaders['train']):.3f}.. "
        f"Validation loss: {valid_loss:.3f}.. "
        f"Validation accuracy: {accuracy:.3f}")

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
    train_model(args, image_datasets=image_datasets, data_loaders=data_loaders)
    # print(f"{args.data_dir} - {args.save_dir} - {args.hidden_units}")





