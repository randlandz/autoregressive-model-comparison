import numpy as np
import torch
from torch.utils.data import TensorDataset

def generate_series(
        length=12000,
        seed=42,
        noise_std=0.05,
        burn_in=200
):
    # x[t] depends on x[t-1], x[t-2] and x[t-10]

    rng = np.random.default_rng(seed)
    total_length = length + burn_in
    series = np.zeros(total_length, dtype=np.float32)

    series[:10] = rng.normal(
        loc=0.0,
        scale=0.5,
        size=10
    )

    for t in range(10, total_length):
        noise = rng.normal(
            loc=0.0,
            scale=noise_std
        )

        series[t] = (
            0.45 * series[t-1]
            - 0.10 * series[t-2]
            + 0.40 * series[t-10]
            + noise
        )

    return series[burn_in:]

def split_series(
        series,
        train_ratio=0.70,
        val_ratio=0.15
):

    # default split: 70% training, 15% validation, 15% test
    
    n = len(series)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = series[:train_end]
    val = series[train_end:val_end]
    test = series[val_end:]

    return train, val, test

def normalize_splits(train, val, test):
    mean = train.mean()
    std = train.std()

    train = (train - mean) / std
    val = (val - mean) / std
    test = (test - mean) / std

    stats = {
        "mean": mean,
        "std": std
    }

    return train, val, test, stats

def create_windows(series, window_size=20):
    X = []
    y = []

    for i in range(len(series) - window_size):
        X.append(
            series[i:i+window_size]
        )
        y.append(series[i+window_size])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    X = torch.tensor(X).unsqueeze(-1)
    y = torch.tensor(y).unsqueeze(-1)

    return X, y

def build_datasets(
        length=12000,
        window_size=20,
        seed=42
):
    series = generate_series(
        length=length,
        seed=seed
    )

    train, val, test = split_series(series)

    train, val, test, stats = normalize_splits(
        train,
        val,
        test
    )

    X_train, y_train = create_windows(
        train,
        window_size
    )

    X_val, y_val = create_windows(
        val,
        window_size
    )

    X_test, y_test = create_windows(
        test,
        window_size
    )

    train_dataset = TensorDataset(
        X_train,
        y_train
    )

    val_dataset = TensorDataset(
        X_val,
        y_val
    )

    test_dataset = TensorDataset(
        X_test,
        y_test
    )

    return(
        train_dataset,
        val_dataset,
        test_dataset,
        stats
    )

if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset, stats = build_datasets()

    X_train, y_train = train_dataset.tensors
    X_val, y_val = val_dataset.tensors
    X_test, y_test = test_dataset.tensors

    print("Train X:", X_train.shape)
    print("Train y:", y_train.shape)

    print("Validation X:", X_val.shape)
    print("Validation y:", y_val.shape)

    print("Test X:", X_test.shape)
    print("Test y:", y_test.shape)

    print("Normalization stats:", stats)