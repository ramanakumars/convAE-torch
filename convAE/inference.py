from .model import *
import os
import glob

class Inference:
    def __init__(self, data, model, batch_size=16):
        self.model = model
        if not isinstance(data, DataGenerator):
            self.data = NumpyGenerator(data, batch_size=batch_size)
        else:
            self.data = data
        self.img_shape  = self.data[0].shape[1:]

    def get_z(self, data_sub=None, batch_size=16):
        if data_sub is None:
            data = self.data
        else:
            if not isinstance(data_sub, DataGenerator):
                data = NumpyGenerator(data_sub, batch_size=batch_size)
            else:
                data = data
        self.model.eval()

        if 'VAE' in self.model.type:
            _, _, ztest = self.model.encode(torch.Tensor(data[0]).to(device))
            _, zshape   = ztest.shape

            z   = np.zeros((data.ndata, zshape))
            mu  = np.zeros((data.ndata, zshape))
            sig = np.zeros((data.ndata, zshape))

            with torch.no_grad():
                for i, x in enumerate(tqdm.tqdm(data)):
                    # account for value and ground truth
                    if len(x) == 2:
                        x = x[0]
                    X = torch.Tensor(x).to(device)
                    mui, sigi, zi = self.model.encode(X)
                    z[i*data.batch_size:(i+1)*data.batch_size]   = zi.cpu().numpy()
                    mu[i*data.batch_size:(i+1)*data.batch_size]  = mui.cpu().numpy()
                    sig[i*data.batch_size:(i+1)*data.batch_size] = sigi.cpu().numpy()

            return mu, sig, z
        elif 'AE' in self.model.type:
            ztest = self.model.encode(torch.Tensor(data[0]).to(device))
            _, zshape   = ztest.shape

            z   = np.zeros((data.ndata, zshape))

            with torch.no_grad():
                for i, x in enumerate(tqdm.tqdm(data)):
                    # account for value and ground truth
                    if len(x) == 2:
                        x = x[0]
                    X = torch.Tensor(x).to(device)
                    zi = self.model.encode(X)
                    z[i*data.batch_size:(i+1)*data.batch_size]   = zi.cpu().numpy()

            return z

    def get_recon(self, data_sub=None, batch_size=16):
        if data_sub is None:
            data = self.data
        else:
            if not isinstance(data_sub, DataGenerator):
                data = NumpyGenerator(data_sub, batch_size=batch_size)
            else:
                data = data

        self.model.eval()

        recon = np.zeros((data.ndata, *self.img_shape))

        if 'DEC' in self.model.type:
            gamma = np.zeros((data.ndata, self.model.n_centroid))

        with torch.no_grad():
            for i, x in enumerate(tqdm.tqdm(data)):
                # account for value and ground truth
                if len(x) == 2:
                    x = x[0]
                X = torch.Tensor(x).to(device)
                if 'DEC' not in self.model.type:
                    recon[i*data.batch_size:(i+1)*data.batch_size] = self.model(X).cpu().numpy()
                else:
                    recon[i*data.batch_size:(i+1)*data.batch_size] = self.model(X)[0].cpu().numpy()
                    gamma[i*data.batch_size:(i+1)*data.batch_size] = self.model(X)[1].cpu().numpy()

        if 'DEC' not in self.model.type:
            return recon
        else:
            return recon, gamma
    
    def get_recon_single(self, x):
        with torch.no_grad():
            X = torch.Tensor(x).to(device)
            if 'DEC' not in self.model.type:
                recon   = self.model(X).cpu().numpy()
            else:
                recon,_ = self.model(X).cpu().numpy()

        return recon


