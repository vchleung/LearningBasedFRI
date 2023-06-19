import torch
import torch.nn as nn
import torch.nn.init as init


class conv(nn.Module):
    def __init__(self, prms):
        super(conv, self).__init__()
        D_in, D_out = prms['N'], prms['K_total']

        if prms['periodic']:
            padding_mode = 'circular'
        else:
            padding_mode = 'zeros'

        self.fc1 = nn.Conv1d(1, 100, 3, stride=1, padding=1, padding_mode=padding_mode)
        self.fc2 = nn.Conv1d(100, 100, 3, stride=1, padding=1, padding_mode=padding_mode)
        self.fc3 = nn.Conv1d(100, 100, 3, stride=1, padding=1, padding_mode=padding_mode)
        self.fc4 = nn.Linear(D_in*100, 100)
        self.fc5 = nn.Linear(100, 100)
        self.fc6 = nn.Linear(100, D_out)
        self.relu = nn.ReLU()

        if prms['sort_tk']:
            self.sort = lambda t_k: torch.sort(t_k)[0]
        else:
            self.sort = lambda t_k: t_k

    def forward(self, x):
        x = x[:, None, :]
        f1 = self.relu(self.fc1(x))
        f2 = self.relu(self.fc2(f1))
        f3 = torch.flatten(self.relu(self.fc3(f2)), start_dim=1)
        f4 = self.relu(self.fc4(f3))
        f5 = self.relu(self.fc5(f4))
        f6 = self.fc6(f5)
        t_k_sorted = self.sort(f6)

        return t_k_sorted

    def _initialize_weights(self):
        init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='conv1d')
        init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='conv1d')
        init.kaiming_uniform_(self.fc3.weight, mode='fan_in', nonlinearity='conv1d')
        init.kaiming_uniform_(self.fc4.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.fc5.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.fc6.weight, mode='fan_in')
