import csv

from pathlib import Path

import matplotlib.pyplot as plt

csv_path = Path("results/metrics/model_comparison.csv")

models = []
parameters = []
best_val_losses = []
test_losses = []

with open(csv_path, newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        models.append(row["model"])
        parameters.append(int(row["parameters"]))
        best_val_losses.append(float(row["best_val_loss"]))
        test_losses.append(float(row["test_loss"]))

def generate_comparison_bar_graph(
        data,
        title,
        ylabel,
        file_name,
        format_decimal=False
):
    plt.figure(figsize=(10, 4))
    bars = plt.bar(models, data)
    if format_decimal:
        plt.bar_label(
            bars,
            labels=[f"{value:.3f}" for value in data]
        )
    else:
        plt.bar_label(bars)

    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel(ylabel)
    plt.tight_layout()

    save_path = Path(f"results/figures/{file_name}")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=150)
    plt.close()

generate_comparison_bar_graph(
    test_losses,
    "Test MSE by Model",
    "Test MSE",
    "test_loss_comparison.png",
    format_decimal=True
)

generate_comparison_bar_graph(
    parameters,
    "Parameter Count by Model",
    "Number of Parameters",
    "parameter_comparison.png"
)