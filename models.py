import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class VAE(nn.Module):
    def __init__(self, nIp, nhid, latent_dim):
        super(VAE, self).__init__()

        ### Encoder layers
        self.enc1 = nn.Linear(nIp, nhid)
        self.bne1 = nn.BatchNorm1d(nhid)
        self.enc2 = nn.Linear(nhid, nhid // 2)
        self.bne2 = nn.BatchNorm1d(nhid // 2)
        self.enc3 = nn.Linear(nhid // 2, nhid // 4)
        self.bne3 = nn.BatchNorm1d(nhid // 4)
        self.enc4 = nn.Linear(nhid // 4, nhid // 4)
        self.bne4 = nn.BatchNorm1d(nhid // 4)
        # Separate the output into mean and variance
        self.enc_mean = nn.Linear(nhid // 4, latent_dim)
        self.enc_log_var = nn.Linear(nhid // 4, latent_dim)

        ### Decoder layers
        self.dec1 = nn.Linear(latent_dim, nhid // 4)
        self.bnd1 = nn.BatchNorm1d(nhid // 4)
        self.dec2 = nn.Linear(nhid // 4, nhid // 4)
        self.bnd2 = nn.BatchNorm1d(nhid // 4)
        self.dec3 = nn.Linear(nhid // 4, nhid // 2)
        self.bnd3 = nn.BatchNorm1d(nhid // 2)
        self.dec4 = nn.Linear(nhid // 2, nhid)
        self.bnd4 = nn.BatchNorm1d(nhid)
        self.dec5 = nn.Linear(nhid, nIp)

    def encode(self, x):
        x = F.gelu(self.enc1(x))
        x = F.gelu(self.enc2(x))
        x = F.gelu(self.enc3(x))
        x = F.gelu(self.enc4(x))
        mean = self.enc_mean(x)
        log_var = self.enc_log_var(x)
        return mean, log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        x = F.gelu(self.dec1(z))
        x = F.gelu(self.dec2(x))
        x = F.gelu(self.dec3(x))
        x = F.gelu(self.dec4(x))
        xHat = torch.sigmoid(self.dec5(x))
        return xHat

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        xHat = self.decode(z)
        return xHat, mean, log_var

    def loss_function(self, recon_x, x, mean, log_var):
        beta_ = 1
        # Reconstruction loss (MSE or BCE, depending on your data)
        # loss = F.mse_loss(recon_x, x)
        loss = F.l1_loss(recon_x, x)
        # KL divergence loss
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return loss+beta_*KLD
    





