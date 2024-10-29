import matplotlib.pyplot as plt
import pickle 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

from util import chain
from tuning import BatchNormalization, GradientClipping

class RecurrentNeuralNetwork(nn.Module):
    def __init__(self, hidden_size, batch_first = True):
        super(RecurrentNeuralNetwork, self).__init__()
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.i2h = nn.Linear(len(alphabets), hidden_size)
        self.h2o = nn.Linear(hidden_size, len(languages))

        # self.batch_norm = BatchNormalization(hidden_size)################
        self.hidden_size = hidden_size
        self.batch_first = batch_first

    def forward(self, x, hidden):
        # x: (batch_size, max_length, len(alphabets))
        hidden = F.tanh(self.i2h(x) + self.h2h(hidden)) # hidden: (batch_size, hidden_size)
        # hidden = self.batch_norm(hidden)#############################
        if self.batch_first:
            output = self.h2o(hidden)
            output = F.log_softmax(output, dim = -1)
        else:
            output = F.log_softmax(self.h2o(hidden), dim = 0)
        # output.shape: batch_size, output_size

        return output, hidden

    def init_hidden(self):
        return torch.zeros(self.hidden_size)

    def train_model(self, train_data, valid_data, epochs = 30, learning_rate = 0.0007,
                    print_every = 1000, accumulation = 2, clip_value=1.0):
        criterion = F.nll_loss
        optimizer = optim.Adam(self.parameters(), lr = learning_rate)
        gradient_clipper = GradientClipping(clip_value)

        step = 0
        train_loss_history = []
        valid_loss_history = []
        for epoch in range(epochs):
            for x, y in train_data:
                step += 1
                # x: (batch_size, max_length, len(alphabets))
                if self.batch_first:
                    x = x.transpose(0, 1)
                # x: (max_length, batch_size, len(alphabets))
                hidden = self.init_hidden()
                for char in x:
                    output, hidden = self(char, hidden)
                loss = criterion(output, y)

                loss = loss / accumulation
                loss.backward()

                gradient_clipper.forward(self)
                # if step % accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()

                mean_loss = torch.mean(loss).item()
                # mean_loss = loss.item() * accumulation

                if step % print_every == 0 or step == 1:
                    train_loss_history.append(mean_loss)
                    valid_loss, valid_acc = self.evaluate(valid_data)
                    valid_loss_history.append(valid_loss)
                    print(f'[Epoch {epoch}, Step {step}] train loss: {mean_loss}, valid loss: {valid_loss}, valid_acc: {valid_acc}')

        return train_loss_history, valid_loss_history

    def evaluate(self, data):
        self.eval()
        criterion = F.nll_loss

        correct, total = 0, 0
        loss_list = []
        with torch.no_grad():
            for x, y in data:
                # x: (batch_size, max_length, len(alphabets))
                if self.batch_first:
                    x = x.transpose(0, 1)
                # x: (max_length, batch_size, len(alphabets))
                hidden = self.init_hidden()
                for char in x:
                    output, hidden = self(char, hidden)
                loss = criterion(output, y)

                loss_list.append(torch.mean(loss).item())
                correct += torch.sum((torch.argmax(output, dim = 1) == y).float())
                total += y.size(0)
            return sum(loss_list) / len(loss_list), correct / total

if __name__ == '__main__':
    import config 
    from data_handler import generate_dataset, modify_dataset_for_ffn, plot_loss_history

    train_dataset, valid_dataset, test_dataset, alphabets, max_length, languages  = generate_dataset()
    
    rnn = RecurrentNeuralNetwork(128)
    train_loss_history, valid_loss_history = rnn.train_model(train_dataset, valid_dataset)

    plot_loss_history(train_loss_history, valid_loss_history)