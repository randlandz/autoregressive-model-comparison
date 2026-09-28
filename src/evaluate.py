import torch

def evaluate_model(
        model,
        test_loader,
        loss_fn,
        device="cpu"
):
    model.to(device)
    
    model.eval()

    total_test_loss = 0

    predictions = []
    targets = []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            predictions.append(pred.cpu())
            targets.append(y.cpu())

            total_test_loss += loss.item()

    predictions = torch.cat(predictions)
    targets = torch.cat(targets)

    avg_test_loss = total_test_loss / len(test_loader)

    return avg_test_loss, predictions, targets