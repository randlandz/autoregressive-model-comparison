import csv
from pathlib import Path


def save_model_results(
        model_name,
        parameters,
        best_epoch,
        best_val_loss,
        test_loss,
        file_path='results/metrics/model_comparison.csv'
):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "model",
        "parameters",
        "best_epoch",
        "best_val_loss",
        "test_loss",
    ]

    new_row = {
        "model": model_name,
        "parameters": parameters,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss
    }

    rows = []

    if file_path.exists():
        with open(file_path, "r", newline="") as file:
            reader = csv.DictReader(file)

            for row in reader:
                rows.append(row)


    updated = False

    for i, row in enumerate(rows):
        if row["model"] == model_name:
            rows[i] = new_row
            updated = True
            break

    if not updated:
        rows.append(new_row)

    with open(file_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns
        )

        writer.writeheader()
        writer.writerows(rows)