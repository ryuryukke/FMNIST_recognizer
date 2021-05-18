# necessary imports
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

def prepare():
    # edit the images so that all the images have the same dimensions and properties
    # convert all the images into tensor, and then normalize
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

    # download the datasets
    trainset = datasets.MNIST(".", download=True,
                            train=True, transform=transform)
    valset = datasets.MNIST(".", download=True,
                            train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                            shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64,
                            shuffle=True)
    
    return trainloader, valloader

def check_data(loader):
    dataiter = iter(loader)
    images, labels = dataiter.next()
    print(images.shape, labels.shape)


def build_model(input_size, hidden_sizes, output_size):
    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                        nn.ReLU(),
                        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                        nn.ReLU(),
                        nn.Linear(hidden_sizes[1], output_size),
                        nn.LogSoftmax(dim=1)
                        )
    return model


def train_model(model, epoch, trainloader):
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    criterion = nn.NLLLoss()
    tik = time()
    for e in range(epoch):
        running_loss = 0
        # traverse all the datasets
        for images, labels in trainloader:
            images = images.view(images.shape[0], -1)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch{e} - Training loss: {running_loss / len(trainloader)}")
    tok = time()
    print(f"training time (in minutes) = {(tok - tik) / 60}")
    return model


def test_model(model, valloader):
    correct_count, all_count = 0, 0
    for images, labels in valloader:
        for i in range(len(labels)):
            img = images[i].view(images[i].shape[0], -1)
            with torch.no_grad():
                logps = model(img)
            ps = torch.exp(logps)
            # print(ps, ps.shape)
            prob = list(ps.numpy()[0])
            pred_label = prob.index(max(prob))
            true_label = labels.numpy()[i]
            if true_label == pred_label:
                correct_count += 1
            all_count += 1
    
    print(f"Number Of Images Tested = {all_count}")
    print(f"Model Accuracy = {correct_count/all_count}")


def main():
    trainloader, valloader = prepare()
    # check_data(trainloader)
    model = build_model(784, [128, 64], 10)
    trained_model = train_model(model, 10, trainloader)
    test_model(trained_model, valloader)
    torch.save(trained_model, "./my_mnist_model.pt")
    

if __name__ == "__main__":
    main()