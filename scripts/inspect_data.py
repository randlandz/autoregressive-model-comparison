from src.data import generate_series
from src.plots import plot_series

from src.data import build_datasets
from src.plots import plot_sample

series = generate_series()

print("Series shape:", series.shape)
print("First 10 values:")
print(series[:10])

plot_series(
    series,
    "results/figures/generated_series.png"
)

print("Figure saved.")

train_dataset, val_dataset, test_dataset, stats = build_datasets()

X_sample, y_sample = train_dataset[0]

print("X sample shape:", X_sample.shape)
print("y sample shape:", y_sample.shape)

plot_sample(
    X_sample,
    y_sample,
    "results/figures/training_example.png"
)

print("Training example figure saved.")