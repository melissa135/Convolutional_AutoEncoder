import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1,5,4,stride=4),
            nn.Tanh(),
            nn.Conv1d(5,10,4,stride=4),
            nn.Tanh(),
            nn.Conv1d(10,5,3,stride=3),
            nn.Tanh()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(5,10,3,stride=3),
            nn.Tanh(),
            nn.ConvTranspose1d(10,5,4,stride=4),
            nn.Tanh(),
            nn.ConvTranspose1d(5,1,4,stride=4)
        )

    def forward(self, x):
        
        x = self.encoder(x)
	x = self.decoder(x)
        return x


class Compression_encoder(nn.Module):
    def __init__(self):
        super(Compression_encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1,5,4,stride=4),
            nn.Tanh(),
            nn.Conv1d(5,10,4,stride=4),
            nn.Tanh(),
            nn.Conv1d(10,5,3,stride=3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        batchsize = x.size()[0]
        x = x.resize(batchsize,5)
        return x
