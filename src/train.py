from torch.utils.data import DataLoader

from src.data import build_datasets

import torch
import torch.nn as nn

import copy

def create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=32
):
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader

def train_model(
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        epochs=50,
        device='cpu'
):
    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = 0
    
    model.to(device)

    for epoch in range(epochs):
        model.train()

        total_train_loss = 0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()

        total_val_loss = 0

        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)

                pred = model(X)
                loss = loss_fn(pred, y)

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_epoch = epoch
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_state)

    print("Best epoch:", best_epoch)
    print("Best val loss:", best_val_loss)

    return train_losses, val_losses, best_epoch, best_val_loss

        

if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset, stats = build_datasets()

    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        32
    )

    X, y = next(iter(train_loader))

    print(
        X.shape,
        y.shape
    )

