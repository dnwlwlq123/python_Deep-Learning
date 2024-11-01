import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from time import time
from typing import List, Dict, Tuple

from conv2d_mine import CNN
from conv2d_torch import SimpleCNN, ResNet
from data_handler import train_data, train_loader, val_loader


def train(
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer_class: torch.optim.Optimizer,
        epochs: int = 50,
        lr: float = 0.0005,  # 학습률 약간 증가
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict[str, List[float]]:
    print(f"Training on {device}")
    model = model.to(device)
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=1e-4)  # weight decay 추가

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_batches = 0
        begin = time()

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping 추가
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == batch_y).sum().item()
            accuracy = correct / batch_y.size(0)

            train_loss += loss.item()
            train_acc += accuracy
            train_batches += 1

            # 메모리 정리
            del outputs, loss
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        avg_train_loss = train_loss / train_batches
        avg_train_acc = train_acc / train_batches

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_batches = 0

        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x, val_y = val_x.to(device), val_y.to(device)

                val_outputs = model(val_x)
                val_batch_loss = criterion(val_outputs, val_y)

                _, predicted = torch.max(val_outputs.data, 1)
                correct = (predicted == val_y).sum().item()
                accuracy = correct / val_y.size(0)

                val_loss += val_batch_loss.item()
                val_acc += accuracy
                val_batches += 1

                del val_outputs, val_batch_loss
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        avg_val_loss = val_loss / val_batches
        avg_val_acc = val_acc / val_batches

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)

        end = time()
        print(f"[Epoch {epoch}/{epochs}] "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Train Acc: {avg_train_acc * 100:.2f}%, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Acc: {avg_val_acc * 100:.2f}%, "
              f"Time: {end - begin:.2f}s")

    return history


def plot_training_history(history: Dict[str, List[float]]):
    """Plot training and validation metrics"""
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(15, 5))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True

    model = ResNet(3, 16, len(train_data.classes))

    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer_class=optim.Adam,
        epochs=50,
        lr=0.0005
    )

    plot_training_history(history)