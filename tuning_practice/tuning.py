import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

class BatchNormalization(nn.Module):
    def __init__(self, hidden_dim, batch_dim = 0):
        super(BatchNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.beta = nn.Parameter(torch.zeros(hidden_dim))
        self.running_mean = torch.zeros(hidden_dim)
        self.running_var = torch.ones(hidden_dim)
        self.momentum = 0.1
        self.eps = 1e-5
        self.batch_dim = 0
        self.momentum = 0.1

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=self.batch_dim, keepdim=True)
            var = x.var(dim=self.batch_dim, keepdim=True, unbiased=False)

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.detach()
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta

class GradientClipping(nn.Module):
    def __init__(self, max_norm):
        super(GradientClipping, self).__init__()
        self.max_norm = max_norm

    def forward(self, model):
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)