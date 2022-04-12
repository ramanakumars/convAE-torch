import torch
from torch import nn
import torchvision
from .io import *
import numpy as np
import tqdm

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

def sample(mu, log_var):
    std = torch.exp(0.5*log_var)
    eps = torch.randn_like(std)
    return mu + eps*std

class Encoder(nn.Module):
    def __init__(self, conv_filt, hidden, input_channels=3):
        super(Encoder, self).__init__()

        self.layers = []
        
        # create the list of convolutional filters
        # to use, so that we can just loop through later on
        conv_filts = [4, 128]
        for i in range(4):
            conv_filts.append(conv_filt)

        # create the convolutional layers
        filt_prev = input_channels
        for filt in conv_filts:
            self.layers.append(nn.Conv2d(in_channels=filt_prev, out_channels=filt, kernel_size=2, stride=1, padding='valid'))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
            self.layers.append(nn.BatchNorm2d(filt))
            filt_prev = filt

        nconv = len(hidden)

        # fully convolutional layers in the bottleneck 
        # instead of a fully connected
        for i in range(nconv):
            self.layers.append(nn.Conv2d(filt, hidden[i], 1, 1, padding='valid'))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm2d(hidden[i]))
            filt = hidden[i]

        self.layers = nn.ModuleList(self.layers)
    
    def forward(self, x):
        # run the input through the layers
        for layer in self.layers:
            x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, conv_filt, hidden, input_channels):
        super(Decoder, self).__init__()
        self.layers = []
        
        filt = input_channels
        # fully convolutional layers in the bottleneck 
        # instead of a fully connected
        nconv = len(hidden)
        for i in range(nconv):
            self.layers.append(nn.Conv2d(filt, hidden[i], 1, 1, padding='valid'))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm2d(hidden[i]))
            filt = hidden[i]
        
        # create the list of convolutional filters
        # to use, so that we can just loop through later on
        conv_filts = []
        for i in range(3):
            conv_filts.append(conv_filt)
        conv_filts.extend([128, 4])

        # create the convolutional layers
        filt_prev = filt
        for j, filt in enumerate(conv_filts):
            self.layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))
            if j==1:
                self.layers.append(nn.ConvTranspose2d(filt_prev, filt, 3, 1, padding=0))
            else:
                self.layers.append(nn.ConvTranspose2d(filt_prev, filt, 2, 1, padding=0))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm2d(filt))
            filt_prev = filt

        self.layers.append(nn.UpsamplingBilinear2d(scale_factor=2))
        self.layers.append(nn.ConvTranspose2d(filt, 3, 2, 1, padding=0))
        self.layers.append(nn.Sigmoid())

        self.layers = nn.ModuleList(self.layers)
    
    def forward(self, x):
        # run the input through the layers
        for layer in self.layers:
            x = layer(x)
        x = torchvision.transforms.functional.crop(x, top=7, left=7, height=384, width=384)
        return x

class DEC(nn.Module):
    def __init__(self, latent_dim, n_centroid, npixels, alpha=1.0, **kwargs):
        super(DEC, self).__init__()

        self.n_centroid = n_centroid
        self.n_centroid = n_centroid
        self.latent_dim = latent_dim
        self.npixels    = npixels
        self.alpha      = 1.0

        self.cluster_centers = nn.Parameter(torch.zeros((self.npixels, self.latent_dim, self.n_centroid)))

    def forward(self, z):
        z = torch.transpose(x, 1, 2)
        z = torch.reshape(x, (self.npixels, self.latent_dim, self.n_centroid))

        q = 1./(1. + torch.sum( torch.square(z - self.cluster_centers), axis=3) / self.alpha)
        q = q**((self.alpha+1.)/2.)
        q = q / torch.sum(q, axis=(1,2), keepdim=True)

        return q


class BaseVAE(nn.Module):
    def __init__(self, conv_filt, hidden, input_channels=3):
        super(BaseVAE, self).__init__()

        self.conv_filt = conv_filt
        self.hidden    = hidden

        self.conv_mu  = nn.Conv2d(hidden[-1], hidden[-1], 1, 1)
        self.conv_sig = nn.Conv2d(hidden[-1], hidden[-1], 1, 1)
        
        self.flat_mu    = nn.Flatten()
        self.flat_sig   = nn.Flatten()

        self.encoder = Encoder(conv_filt, hidden, input_channels)
        self.decoder = Decoder(conv_filt, hidden[::-1], hidden[-1])

        self.type = ['VAE']

    def encode(self, x):
        enc = self.encoder(x)
        
        mu      = self.flat_mu(self.conv_mu(enc))
        log_var = self.flat_sig(self.conv_sig(enc))
        z       = sample(mu, log_var)

        return mu, log_var, z

    def decode(self, z):
        dec_inp = torch.reshape(z, (z.shape[0], self.hidden[-1], 5, 5))

        dec = self.decoder(dec_inp)

        return dec

    def forward(self, x):
        out = self.decode(self.encode(x)[2])

        return out

class BaseAE(nn.Module):
    def __init__(self, conv_filt, hidden, input_channels=3):
        super(BaseVAE, self).__init__()

        self.conv_filt = conv_filt
        self.hidden    = hidden
        
        self.flat_z   = nn.Flatten()

        self.encoder = Encoder(conv_filt, hidden, input_channels)
        self.decoder = Decoder(conv_filt, hidden[::-1], hidden[-1])

        self.type = ['AE']

    def encode(self, x):
        enc = self.encoder(x)
        
        z   = self.flat_z(enc)

        return z

    def decode(self, z):
        dec_inp = torch.reshape(z, (z.shape[0], self.hidden[-1], 5, 5))

        dec = self.decoder(dec_inp)

        return dec

    def forward(self, x):
        out = self.decode(self.encode(x))

        return out

class DECAE(nn.Module):
    def __init__(self, conv_filt, hidden, n_centroid=10, input_channels=3):
        super(BaseVAE, self).__init__()

        self.conv_filt  = conv_filt
        self.hidden     = hidden
        self.n_centroid = n_centroid
        
        self.flat_z   = nn.Flatten()

        self.encoder = Encoder(conv_filt, hidden, input_channels)
        self.decoder = Decoder(conv_filt, hidden[::-1], hidden[-1])
        self.DEC     = DEC(self.hidden[-1], n_centroid, 25)

        self.type = ['DEC','AE']

    def encode(self, x):
        enc = self.encoder(x)
        
        z   = self.flat_z(enc)

        return z

    def decode(self, z):
        dec_inp = torch.reshape(z, (z.shape[0], self.hidden[-1], 5, 5))

        dec = self.decoder(dec_inp)

        return dec


    def forward(self, x):
        z   = self.encode(x)
        out = self.decode(z)
        q   = self.DEC(z)

        return out, q

