import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, List, Callable

from data_handler import LanguagePair, Vocabulary
from transformer_layers import Transformer

class Trainer:
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 save_dir: str = 'models/'
                 ):
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer


    def train(self,
              train_data: DataLoader,
              valid_data: Optional[DataLoader] = None,
              num_epochs: int = 10,
              print_every: int = 1,
              evaluate_every: int = 1,
              evaluate_metric: List[Callable] = [],
              source_vocab: Vocabulary = Vocabulary(),
              target_vocab: Vocabulary = Vocabulary(),
              ) -> List[float]:

        loss_history = []
        for epoch in range(1, 1 + num_epochs):
            self.model.train()
            epoch_train_loss = 0

            for batch_idx, (src, tgt) in enumerate(train_data):
                output = ...
                loss = self.criterion(tgt, output)
                epoch_train_loss += loss.item()

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            if valid_list is not None and epoch:
                evaluate_every == 0:
                valid_loss, valid_metric = self.evaluate



if __name__ == '__main__':
    import config
    trainer = Trainer(..., device = config.device)