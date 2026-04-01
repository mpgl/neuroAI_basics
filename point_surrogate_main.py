import pickle
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

sys.path.append(str(Path(__file__).parent.parent))

from models import CUBAPointSurrogateNet
from utils import clean_dataset, plot_results, train_network

# Default hyperparameters & pipeline parameters ################################

@dataclass
class Config:
    # Simulation
    DT: float = 1e-3 # time step in seconds
    TIME: float = 30e-3 # image presentation time in seconds
    SEED: int = 42
    DEVICE: str = "cpu"

    # Neuron properties
    THRESHOLD: float = 1.0
    RESET: float = 0.0
    R: float = 5
    TAU_MEM: float = 10e-3 # membrane time constant in seconds
    TAU_SYN: float = 2e-3 # synaptic time constant in seconds

    # Network
    NUM_INPUTS: int = 784
    NUM_HIDDEN: int = 32
    NUM_OUTPUTS: int = 10
    INPUT_SCALE: float = 1.0 # max probability of input spikes

    # Training
    DATASET: str = 'MNIST'
    BATCH_SIZE: int = 128
    EPOCHS: int = 3
    OPTIMIZER: str = 'SGD'
    LR: float = 1e-4

    # Pipeline
    EXPERIMENT: str = 'test'
    WORKERS: int = 4
    PRINT_PROGRESS: int = 1 # if 1, print training progress

# Main simulation function #####################################################
def main():
    c = Config()
    device = torch.device(c.DEVICE)
    torch.manual_seed(c.SEED)

    # Create the network
    num_steps = int(c.TIME / c.DT)
    network = CUBAPointSurrogateNet(
        num_inputs=c.NUM_INPUTS,
        num_hidden=c.NUM_HIDDEN,
        num_outputs=c.NUM_OUTPUTS,
        num_steps=num_steps,
        threshold=c.THRESHOLD,
        reset=c.RESET,
        R=c.R,
        tau_mem=c.TAU_MEM,
        tau_syn=c.TAU_SYN,
        dt=c.DT)

    # Load and prepare the dataset
    project_root = Path(__file__).parent.parent
    datasets_dir = project_root / "datasets"

    train_data = getattr(datasets, c.DATASET)(
        root=str(datasets_dir),
        train=True,
        download=True,
        transform=ToTensor())

    dt_input_scale = c.INPUT_SCALE * (c.DT / 1e-3) # scale down bernoulli prob when using smaller time steps
    train_data = clean_dataset(train_data, scale=dt_input_scale)

    test_data = getattr(datasets, c.DATASET)(
        root=str(datasets_dir),
        train=False,
        download=True,
        transform=ToTensor())

    test_data = clean_dataset(test_data, scale=dt_input_scale)

    # Train and evaluate the network
    t_start = time.time()
    print(f"Training started: {c.EXPERIMENT}\n")
    results = train_network(
        network,
        train_data,
        test_data,
        batch_size=c.BATCH_SIZE,
        epochs=c.EPOCHS,
        optimizer_name=c.OPTIMIZER,
        lr=c.LR,
        device=device,
        print_progress=bool(c.PRINT_PROGRESS),
        num_workers=c.WORKERS,
        pin_memory=True if c.DEVICE == "cuda" else False)
    t_end = time.time()

    # Report results
    best_test_acc = max(results['test_acc'])
    best_epoch = results['test_acc'].index(best_test_acc)
    print(f"\nBest test acc: {best_test_acc:.2f}% at epoch {best_epoch + 1}")

    dur = t_end - t_start
    print(f"Training ended: {c.EXPERIMENT} [Time = {dur:.2f}s]\n")

    # Save results and config
    output_dir = Path("outputs") / c.EXPERIMENT
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "results.pkl"
    to_save = {
        'train_losses': results['train_loss'],
        'test_losses': results['test_loss'],
        'test_accuracies': results['test_acc'],
        'config': asdict(c),
    }

    with open(results_file, 'wb') as f:
        pickle.dump(to_save, f)

    plot_results(results['train_loss'], results['test_loss'], results['test_acc'])


if __name__ == "__main__":
    main()
