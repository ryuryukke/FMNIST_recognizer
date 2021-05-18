import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset, SubsetRandomSampler
from torchvision import datasets, transforms
from sklearn.model_selection import KFold
import pandas as pd
import argparse
import subprocess
from model import MLP, CNN


def setup_parser():
    parser = argparse.ArgumentParser(add_help=False)
    # model setting
    parser.add_argument("--model", type=str, default="MLP",
                        help="You can choose a model out of 2 models, 'MLP', 'CNN'")
    parser.add_argument("--batch_size", type=int, default=64)
    # for MLP model
    parser.add_argument("--fc1", type=int, default=512)
    parser.add_argument("--fc2", type=int, default=128)
    # train setting
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--k_folds", type=int, default=5)
    # parser.add_argument("--train_log", type=str, default="./train_log.csv")
    # parser.add_argument("--test_log", type=str, default="./test_log.csv")
    # GPU setting
    # parser.add_argument("--gpus", type=int, default=0)
    return parser


def create_k_splitted_data_loaders(args):
    kfolds_dataloader = []
    train_data = datasets.FashionMNIST(
        # if FashionMNIST does't exist in a path, "root", download it.
        root=".",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])
    )
    test_data = datasets.FashionMNIST(
        root=".",
        train=False,
        download=True,
        transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])
    )

    dataset = ConcatDataset([train_data, test_data])
    kfold = KFold(n_splits=args.k_folds, shuffle=True)
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        train_subsampler, test_subsampler = SubsetRandomSampler(train_ids), SubsetRandomSampler(test_ids)
        train_loader = DataLoader(
                                dataset,
                                batch_size=args.batch_size,
                                sampler=train_subsampler
                                )
        test_loader = DataLoader(
                                dataset,
                                batch_size=args.batch_size,
                                sampler=test_subsampler
                                )
        kfolds_dataloader.append((fold + 1, train_loader, test_loader))

    return kfolds_dataloader


def reset_weights(model):
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            print(f"Reset trainable parameters of layer:\n{layer}")
            layer.reset_parameters()


def train_loop(dataloader, model, args):
    size = len(dataloader.dataset)
    print(f"size: {size}")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    log = []

    for epoch in range(args.epochs):
        current_loss = 0
        print(f"Epoch {epoch + 1}\n------------------")
        for batch, (X, y) in enumerate(dataloader):
            pred = model(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_loss += loss.item()
            if batch % 100 == 99:
                print(f"Loss after {batch + 1} mini-batches: {current_loss / 100:.3g}")
        log.append(current_loss / (batch + 1))
    
    return model, log


def test_loop(dataloader, model, fold):
    size = len(dataloader.dataset)
    loss_fn = nn.CrossEntropyLoss()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
    test_loss /= size
    correct_rate = (correct / size) * 100
    print(f"Accuracy for fold {fold}: {correct_rate:.3g}")
    print(f"Test loss for fold {fold}: {test_loss:.3g}")
    return correct_rate


def main():
    parser = setup_parser()
    args = parser.parse_args()
    subprocess.run(f"mkdir {args.model}", shell=True)
    torch.manual_seed(42)
    # device="cuda" if args.gpus == -1 else "cpu"

    k_data_loaders = create_k_splitted_data_loaders(args)
    if args.model == "MLP":
        model = MLP(args)
    elif args.model == "CNN":
        model = CNN(args)
    model.apply(reset_weights)
    
    acc_results, logs = [], []
    for fold, train_loader, test_loader in k_data_loaders:
        print(f"FOLD {fold}\n-----------------------------")
        print("Starting training...")
        model, log = train_loop(train_loader, model, args)
        logs.append(log)

        print("Training process has finished. Saving trained model.")
        torch.save(model.state_dict(), f"./{args.model}/model_fold_{fold}.pth")

        print("Starting testing...")
        correct_rate = test_loop(test_loader, model, fold)
        acc_results.append(correct_rate)

        print("Resetting the model weights...")
        reset_weights(model)

    print(f"K-FOLD CROSS VALIDATION RESULTS FOR {args.k_folds} FOLDS\n----------------------")
    print(f"Average: {sum(acc_results) / len(acc_results):.3g}%")


if __name__ == "__main__":
    main()