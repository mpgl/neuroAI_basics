import time

import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import mplcyberpunk


def train_network(
        model: nn.Module,
        train_data: torch.utils.data.Dataset, # type: ignore
        test_data: torch.utils.data.Dataset, # type: ignore
        batch_size: int,
        epochs: int,
        optimizer_name: str,
        lr: float,
        device: torch.device,
        print_progress: bool = True,
        pin_memory: bool = False,
        num_workers: int = 0):

    # Create DataLoaders
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
        multiprocessing_context='fork' if num_workers > 0 else None)

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        multiprocessing_context='fork' if num_workers > 0 else None)

    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer_params = {'params': model.parameters(), 'lr': lr}
    if optimizer_name == 'SGD':
        optimizer_params.update({'momentum': 0.9, 'weight_decay': 0.0})
    optimizer = getattr(torch.optim, optimizer_name)(**optimizer_params)

    # Main training and evaluation loop
    model.to(device)
    results = {'train_loss': [], 'test_loss': [], 'test_acc': []}

    # ---- Training Phase -------------------------------------------------#
    for epoch in range(epochs):
        epoch_start_time = time.time()

        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        results['train_loss'].append(train_loss)

        # ---- Testing Phase --------------------------------------------------#
        model.eval()
        test_loss, test_acc = 0, 0
        with torch.no_grad():
            for X_test, y_test in test_loader:
                X_test, y_test = X_test.to(device), y_test.to(device)
                test_pred = model(X_test)
                test_loss += loss_fn(test_pred, y_test).item()
                test_acc += (test_pred.argmax(dim=1) == y_test).sum().item()
        test_loss /= len(test_loader)
        test_acc = (test_acc / len(test_data)) * 100
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)

        # ---- Progress Printing ----------------------------------------------#
        epoch_duration = time.time() - epoch_start_time
        if print_progress:
            print(
                f"Epoch: {epoch+1} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Test Loss: {test_loss:.4f} | "
                f"Test Acc: {test_acc:.2f}% | "
                f"Time: {epoch_duration:.2f}s")

    return results


def clean_dataset(dataset: torch.utils.data.Dataset,
                  max_devide: bool = True,
                  scale: float = 1.0
                  ) -> torch.utils.data.TensorDataset:
    """
    Extracts data and targets from the dataset and normalizes the data. It can also significantly
    improve simulation performance.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The dataset to be cleaned.

    max_devide : bool, optional
        If True, the data will be normalized by dividing it by its maximum value.
        Default is False.

    scale : float, optional
        A scaling factor to apply to the data. Default is 1.0.

    Returns
    -------
    torch.utils.data.TensorDataset
        The cleaned dataset.
    """
    X, y = dataset.data.unsqueeze(1).float(), dataset.targets.long()
    if max_devide:
        X /= X.max()
    X *= scale  # Scale the data
    return torch.utils.data.TensorDataset(X, y)


def plot_results(
        train_losses,
        test_losses,
        test_accuracies,
        save=False):
    """Plot training and test losses along with test accuracies."""
    # Ensure all inputs are on CPU if they are tensors
    if torch.is_tensor(train_losses):
        train_losses = train_losses.cpu().numpy()
    if torch.is_tensor(test_losses):
        test_losses = test_losses.cpu().numpy()
    if torch.is_tensor(test_accuracies):
        test_accuracies = test_accuracies.cpu().numpy()

    with plt.style.context("cyberpunk"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Left panel: Train and test loss
        ax1.plot(train_losses, label='Train')
        ax1.plot(test_losses, label='Test')
        ax1.set_xlabel('Epoch', color='white')
        ax1.set_ylabel('Loss', color='white')
        ax1.set_title('Loss Curves', color='white')
        ax1.tick_params(colors='white')
        ax1.legend()
        mplcyberpunk.add_glow_effects(ax1)

        # Right panel: Test accuracy
        ax2.plot(test_accuracies, label='Test', color='#00ff41')
        ax2.set_xlabel('Epoch', color='white')
        ax2.set_ylabel('Accuracy (%)', color='white')
        ax2.set_title('Accuracy Curves', color='white')
        ax2.tick_params(colors='white')
        ax2.legend()
        mplcyberpunk.add_glow_effects(ax2)

        fig.tight_layout()
        plt.show()
        if save:
            fig.savefig("training_results.png", dpi=300)
