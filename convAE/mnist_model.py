from .model import *

class MNISTDecoder(nn.Module):
    def __init__(self, conv_filt, hidden, input_channels, conv_filts=[4, 128], n_upsample=3):
        super(MNISTDecoder, self).__init__()
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
        for i in range(n_upsample):
            conv_filts.append(conv_filt)
        conv_filts = conv_filts[::-1]

        # create the convolutional layers
        filt_prev = filt
        for j, filt in enumerate(conv_filts):
            self.layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))
            if j==1:
                self.layers.append(nn.ConvTranspose2d(filt_prev, filt, 3, 1, padding=0))
            else:
                self.layers.append(nn.ConvTranspose2d(filt_prev, filt, 3, 1, padding=0))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm2d(filt))
            filt_prev = filt

        self.layers.append(nn.UpsamplingBilinear2d(scale_factor=2))
        self.layers.append(nn.ConvTranspose2d(filt, 1, 2, 1, padding=0))
        self.layers.append(nn.Sigmoid())

        self.layers = nn.ModuleList(self.layers)
    
    def forward(self, x):
        # run the input through the layers
        for layer in self.layers:
            x = layer(x)
        x = torchvision.transforms.functional.crop(x, top=1, left=1, height=28, width=28)
        return x

class MNISTDECAE(nn.Module):
    def __init__(self, conv_filt, hidden, n_centroid=10, input_channels=1):
        super(MNISTDECAE, self).__init__()

        self.conv_filt  = conv_filt
        self.hidden     = hidden
        self.n_centroid = n_centroid
        self.latent_dim = hidden[-1]*4
        
        self.flat_z   = nn.Flatten()

        self.encoder = Encoder(conv_filt, hidden, input_channels, n_downsample=2, conv_filts=[4])
        self.decoder = MNISTDecoder(conv_filt, hidden[::-1], hidden[-1], n_upsample=1, conv_filts=[4])
        self.DEC     = DEC(self.latent_dim, n_centroid)

        self.type = ['DEC','AE']

    def encode(self, x):
        enc = self.encoder(x)
        
        z   = self.flat_z(enc)

        return z

    def decode(self, z):
        dec_inp = torch.reshape(z, (z.shape[0], self.hidden[-1], 2, 2))

        dec = self.decoder(dec_inp)

        return dec


    def forward(self, x):
        z   = self.encode(x)
        q   = self.DEC(z)
        out = self.decode(z)

        return out, q
