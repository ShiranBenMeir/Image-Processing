import math
import os
import torch
import torch.nn as nn
import torch.utils
import torch.distributions
import torchvision
import torch.nn.functional as F

IMGSIZE = 64
criterion = nn.MSELoss()
latent_dims= 5

class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(IMGSIZE*IMGSIZE, 30)
        self.linear2 = nn.Linear(30, latent_dims)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return self.linear2(x)

class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 30)
        self.linear2 = nn.Linear(30, IMGSIZE*IMGSIZE)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, IMGSIZE, IMGSIZE))

class Autoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def train(autoencoder, trainloader, epochs=15):
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in range(epochs):
        tot_loss = 0
        for x, y in trainloader:
            opt.zero_grad()
            x = x.view(-1, 1, IMGSIZE * IMGSIZE)
            x_hat = autoencoder(x)
            x_hat = x_hat.view(-1, 1, IMGSIZE * IMGSIZE)
            loss = criterion(x_hat,x)
            loss.backward()
            opt.step()
            tot_loss += loss.item()
        tot_loss = tot_loss / len(trainloader)
        print('auto train: ',tot_loss)
    return autoencoder



def val(autoencoder, test_loader):
    preds = []
    labels = []
    with torch.no_grad():
        tot_loss = 0
        for x, y in test_loader:
            x = x.view(-1, 1, IMGSIZE * IMGSIZE)
            x_hat = autoencoder(x)
            x_hat = x_hat.view(-1, 1, IMGSIZE * IMGSIZE)
            loss = criterion(x_hat, x)
            tot_loss += loss.item()
    return tot_loss

transform_d = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(IMGSIZE),
                                              torchvision.transforms.Grayscale(1),
                                              torchvision.transforms.ToTensor()])
#train_path = os.getcwd() + "\\cell_images2"





# --------------------------------------------------------------------------------------------


