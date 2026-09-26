from pathlib import Path

import matplotlib.pyplot as plt

def plot_series(series, save_path, num_steps=300):
    plt.figure(figsize=(10, 4))

    plt.plot(series[:num_steps])

    plt.title("Generated Autoregressive Time Series")
    plt.xlabel("Time step")
    plt.ylabel("Value")

    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_sample(input_window, target, save_path):
    input_window = input_window.squeeze().numpy()
    target = target.item()

    input_steps = range(len(input_window))
    target_step = len(input_window)

    plt.figure(figsize=(10, 4))

    plt.plot(
        input_steps,
        input_window,
        marker="o",
        label="Input window"
    )

    plt.scatter(
        target_step,
        target,
        s=80,
        label="Target"
    )

    plt.axvline(
        target_step - 0.5,
        linestyle='--',
        alpha=0.5
    )

    plt.title("One Autoregressive Training Example")
    plt.xlabel("Relative Time Step")
    plt.ylabel("Normalized Value")
    plt.legend()

    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=150)
    plt.close()