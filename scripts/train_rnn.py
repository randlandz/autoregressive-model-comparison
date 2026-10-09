import torch
import torch.nn as nn

from src.data import build_datasets
from src.train import create_dataloaders, train_model   
from src.models.rnn import RNN

from src.plots import plot_loss_curves
from src.plots import plot_predictions_vs_targets

from src.evaluate import evaluate_model

from src.utils import set_seed

from src.results import save_model_results

set_seed(42)

train_dataset, val_dataset, test_dataset, stats = build_datasets()

train_loader, val_loader, test_loader = create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size=32
)

model = RNN(1, 64)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)

train_losses, val_losses, best_epoch, best_val_loss = train_model(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    epochs=30,
    device=device
)

plot_loss_curves(
    train_losses,
    val_losses,
    model_name="RNN",
    save_path="results/figures/rnn_loss.png"
)

avg_test_loss, predictions, targets = evaluate_model(
    model,
    test_loader,
    loss_fn,
    device
)

print("Test loss:", avg_test_loss)
print("Predictions shape:", predictions.shape)
print("Targets shape:", targets.shape)

parameter_count = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    "Parameters:",
    parameter_count
)


plot_predictions_vs_targets(
    predictions,
    targets,
    model_name="RNN",
    save_path="results/figures/rnn_predictions.png"
)

save_model_results(
    model_name="RNN",
    parameters=parameter_count,
    best_epoch=best_epoch,
    best_val_loss=best_val_loss,
    test_loss=avg_test_loss
)