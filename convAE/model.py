import torch
from torch import nn
import torchvision
from torchinfo import summary
from .io import *
import numpy as np
import tqdm

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

def sample(mu, log_var):
    std = torch.exp(0.5*log_var)
    eps = torch.randn_like(std)
    return mu + eps*std

class Downsample(nn.Module):
    def __init__(self, conv_filt, conv_filts, input_channels):
        super(Downsample, self).__init__()

        # create the convolutional layers
        filt_prev = input_channels
        self.layers = []
        for filt in conv_filts:
            self.layers.append(nn.Conv2d(in_channels=filt_prev, out_channels=filt, kernel_size=3,
                                         stride=1, padding='valid'))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            self.layers.append(nn.BatchNorm2d(filt))
            filt_prev = filt
            
        #self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        self.layers = nn.ModuleList(self.layers)
    
    def forward(self, x):
        # run the input through the layers
        for layer in self.layers:
            x = layer(x)
        return x

class Bottleneck(nn.Module):
    def __init__(self, hidden, conv_filt):
        super(Bottleneck, self).__init__()
        nconv = len(hidden)
        
        filt = conv_filt

        # fully convolutional layers in the bottleneck 
        # instead of a fully connected
        self.layers = []
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

class BottleneckRes(nn.Module):
    def __init__(self, hidden, conv_filt, n_rep=3):
        super(BottleneckRes, self).__init__()
        nconv = len(hidden)
        
        filt = conv_filt

        self.layers = []

        self.n_rep = n_rep

        # fully convolutional layers in the bottleneck 
        # instead of a fully connected
        for i in range(nconv):
            for j in range(self.n_rep):
                self.layers.append(nn.Conv2d(filt, hidden[i], 1, 1, padding='valid'))
                self.layers.append(nn.ReLU())
                self.layers.append(nn.BatchNorm2d(hidden[i]))
                filt = hidden[i]
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for i in range(len(self.layers)//(3*self.n_rep)):
            out0 = self.layers[i*self.n_rep*3  ](x)    # conv2d
            out0 = self.layers[i*self.n_rep*3+1](out0) # activation
            out0 = self.layers[i*self.n_rep*3+2](out0) # batchnorm

            for j in range(1, self.n_rep):
                ind = (i*self.n_rep+j)*3
                if j==1:
                    out = self.layers[ind  ](out0) # conv2d
                else:
                    out = self.layers[ind  ](out) # conv2d
                out     = self.layers[ind+1](out) # activation
                out     = self.layers[ind+2](out) # batchnorm
            x = torch.add(out, out0)

        return x

class Encoder(nn.Module):
    '''
        Encoder module that encodes the input image into a latent vector
        Input:  ([batch_size], n_channels, x, x) dimensional image
        Output: ([batch_size], n_latent_dim) output vector
        n_latent_dim is determined by number of downsampling layers and value of x
    '''
    def __init__(self, conv_filt, hidden, input_channels=3, conv_filts = [8, 64], n_downsample=4, resnet=False):
        super(Encoder, self).__init__()

        self.layers = []
        
        # create the list of convolutional filters
        # to use, so that we can just loop through later on
        for i in range(n_downsample):
            conv_filts.append(conv_filt)
        
        self.downsample = Downsample(conv_filt, conv_filts, input_channels)

        if resnet:
            self.bottleneck = BottleneckRes(hidden, conv_filt)
        else:
            self.bottleneck = Bottleneck(hidden, conv_filt)

    def forward(self, x):
        # run the input through the layers
        x = self.downsample(x)
        x = self.bottleneck(x)
        return x

class Decoder(nn.Module):
    ''' 
        Decoder module that recreates the image based on the latent vector
        Input:  ([batch_size], n_latent, y, y)  reshaped latent vecotr
        Output: ([batch_size], 3, x, x) image
        This decoder is tuned for a (n_latent, 5, 5) input image -> (5, 584, 384) output image
    '''
    def __init__(self, conv_filt, hidden, input_channels, conv_filts=[8, 64], n_upsample=3):
        super(Decoder, self).__init__()
        self.layers = []
        
        filt = input_channels
        # fully convolutional layers in the bottleneck 
        # instead of a fully connected
        nconv = len(hidden)
        for i in range(nconv):
            self.layers.append(nn.Conv2d(filt, hidden[i], 1, 1, padding='valid'))
            self.layers.append(nn.ReLU())
            #self.layers.append(nn.BatchNorm2d(hidden[i]))
            filt = hidden[i]
        
        # create the list of convolutional filters
        # to use, so that we can just loop through later on
        for i in range(n_upsample):
            conv_filts.append(conv_filt)
        conv_filts = conv_filts[::-1]

        #self.layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))

        # create the convolutional layers
        filt_prev = filt
        for j, filt in enumerate(conv_filts):
            #self.layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))
            if j==len(conv_filts)-1:
                self.layers.append(nn.ConvTranspose2d(filt_prev, filt, 5, stride=2, padding=0))
            else:
                self.layers.append(nn.ConvTranspose2d(filt_prev, filt, 3, stride=2, padding=0))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm2d(filt))
            filt_prev = filt

        #self.layers.append(nn.UpsamplingBilinear2d(scale_factor=2))
        self.layers.append(nn.ConvTranspose2d(filt, 3, 3, stride=2, padding=0, dilation=2))
        self.layers.append(nn.Sigmoid())

        self.layers = nn.ModuleList(self.layers)
    
    def forward(self, x):
        # run the input through the layers
        for layer in self.layers:
            x = layer(x)
        x = torchvision.transforms.functional.crop(x, top=4, left=4, height=384, width=384)
        return x

class DecoderUpSample(nn.Module):
    ''' 
        Decoder module that recreates the image based on the latent vector
        Input:  ([batch_size], n_latent, y, y)  reshaped latent vecotr
        Output: ([batch_size], 3, x, x) image
        This decoder is tuned for a (n_latent, 5, 5) input image -> (5, 584, 384) output image
    '''
    def __init__(self, conv_filt, hidden, input_channels, conv_filts=[8, 64], n_upsample=3):
        super(DecoderUpSample, self).__init__()
        self.layers = []
        
        filt = input_channels
        # fully convolutional layers in the bottleneck 
        # instead of a fully connected
        nconv = len(hidden)
        for i in range(nconv):
            self.layers.append(nn.Conv2d(filt, hidden[i], 1, 1, padding='valid'))
            self.layers.append(nn.ReLU())
            #self.layers.append(nn.BatchNorm2d(hidden[i]))
            filt = hidden[i]
        
        # create the list of convolutional filters
        # to use, so that we can just loop through later on
        for i in range(n_upsample):
            conv_filts.append(conv_filt)
        conv_filts = conv_filts[::-1]

        #self.layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))

        # create the convolutional layers
        filt_prev = filt
        for j, filt in enumerate(conv_filts):
            self.layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))
            if j==1:
                self.layers.append(nn.ConvTranspose2d(filt_prev, filt, 3, padding=1))
            else:
                self.layers.append(nn.ConvTranspose2d(filt_prev, filt, 3, padding=0))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm2d(filt))
            filt_prev = filt

        self.layers.append(nn.UpsamplingBilinear2d(scale_factor=2))
        self.layers.append(nn.ConvTranspose2d(filt, 3, 3, padding=0))
        self.layers.append(nn.Sigmoid())

        self.layers = nn.ModuleList(self.layers)
    
    def forward(self, x):
        # run the input through the layers
        for layer in self.layers:
            x = layer(x)
        x = torchvision.transforms.functional.crop(x, top=15, left=15, height=384, width=384)
        return x

class DEC(nn.Module):
    ''' 
        Deep-Embedding Clustering layer. Calculates the cluster probability for a given
        input latent vector based on the Student t-distribution
        See: https://arxiv.org/abs/1511.06335
    '''
    def __init__(self, latent_dim, n_centroid, alpha=1.0, **kwargs):
        super(DEC, self).__init__()

        self.n_centroid = n_centroid
        self.n_centroid = n_centroid
        self.latent_dim = latent_dim
        self.alpha      = alpha
        self.power      = (alpha+1.)/2.

        clust_centers = nn.init.xavier_uniform_(torch.zeros(self.n_centroid, self.latent_dim, dtype=torch.float))
        self.cluster_centers = nn.Parameter(clust_centers, requires_grad=True)

    def forward(self, z):
        #z = torch.transpose(z, 1, 2)
        q1 = 1./(1. + torch.sum( (z.unsqueeze(1) - self.cluster_centers)**2., 2 ) / self.alpha)
        q2 = q1**self.power
        q  = q2 / torch.sum(q2, dim=1, keepdim=True)

        return q

class BaseVAE(nn.Module):
    '''
        Simple Convolutional VAE network that joins the Encoder
        and Decoder module from above. 
    '''
    def __init__(self, conv_filt, hidden, input_channels=3, resnet=False):
        super(BaseVAE, self).__init__()

        self.conv_filt = conv_filt
        self.hidden    = hidden

        self.conv_mu  = nn.Conv2d(hidden[-1], hidden[-1], 1, 1)
        self.conv_sig = nn.Conv2d(hidden[-1], hidden[-1], 1, 1)
        
        self.flat_mu    = nn.Flatten()
        self.flat_sig   = nn.Flatten()

        self.encoder = Encoder(conv_filt, hidden, input_channels, resnet=resnet, n_downsample=4)
        self.decoder = DecoderUpSample(conv_filt, hidden[::-1], hidden[-1], n_upsample=3)

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
    '''
        Simple Convolutional AE network that joins the Encoder
        and Decoder module from above, without the sampling layer used
        by the VAE
    '''
    def __init__(self, conv_filt, hidden, input_channels=3, resnet=False, upsample=True):
        super(BaseAE, self).__init__()

        self.conv_filt  = conv_filt
        self.hidden     = hidden
        self.latent_dim = hidden[-1]*25
        
        self.flat_z   = nn.Flatten()

        self.encoder = Encoder(conv_filt, hidden, input_channels, resnet=resnet, n_downsample=4)
        if upsample:
            self.decoder = DecoderUpSample(conv_filt, hidden[::-1], hidden[-1], n_upsample=3)
        else:
            self.decoder = Decoder(conv_filt, hidden[::-1], hidden[-1], n_upsample=3)


        self.type = ['AE']

    def encode(self, x):
        enc = self.encoder(x)
        
        z   = self.flat_z(enc)

        return z

    def decode(self, z):
        dec_inp = torch.reshape(z, (z.shape[0], self.hidden[-1], 4, 4))

        dec = self.decoder(dec_inp)

        return dec

    def forward(self, x):
        out = self.decode(self.encode(x))

        return out

class DECAE(nn.Module):
    '''
        An AE that has the DEC component applied to the latent vectors
    '''
    def __init__(self, conv_filt, hidden, n_centroid=10, input_channels=3):
        super(DECAE, self).__init__()

        self.conv_filt  = conv_filt
        self.hidden     = hidden
        self.n_centroid = n_centroid
        self.latent_dim = hidden[-1]*25
        
        self.flat_z   = nn.Flatten()

        self.encoder = Encoder(conv_filt, hidden, input_channels, n_downsample=3)
        self.decoder = Decoder(conv_filt, hidden[::-1], hidden[-1], n_upsample=4)
        self.DEC     = DEC(self.latent_dim, n_centroid)

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

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layers = []

        filts = [3, 6, 8, 16, 16, 16, 16]

        for i in range(6):
            self.layers.append(nn.Conv2d(filts[i], filts[i+1], 3, stride=1, padding=0))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool2d(2))
            self.layers.append(nn.BatchNorm2d(filts[i+1]))

        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(256, 64))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(64))
        self.layers.append(nn.Dropout(0.2))
        self.layers.append(nn.Linear(64, 16))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(16))
        self.layers.append(nn.Dropout(0.2))
        self.layers.append(nn.Linear(16, 1))
        self.layers.append(nn.Sigmoid())
        self.layers = nn.ModuleList(self.layers)
    
    def forward(self, x):
        # run the input through the layers
        for layer in self.layers:
            x = layer(x)
        return x

