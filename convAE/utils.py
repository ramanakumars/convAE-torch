from .model import *
import os

BATCH_PRINT = 5

def mse_loss(y_true, y_pred):
    mse = torch.mean(torch.sum(torch.square(y_true - y_pred), axis=(-1, -2)), axis=-1)
    return torch.mean(mse)

def kl_loss(mu, sig, mup, sig0=-4):
    kl = 0.5*torch.mean(-1 - sig + sig0 + (torch.square(mu-mup) + torch.exp(sig))/np.exp(sig0), axis=-1)
    return torch.mean(kl)

def train_VAE_batch(data, model, optimizer, kl_beta=0.1):
    size = len(data)
    model.train()
    pbar = tqdm.tqdm(data)

    print_loss, print_mse, print_kl = 0, 0, 0
    for batch, x in enumerate(pbar):
        X = torch.Tensor(x).to(device)

        # Compute prediction error
        mu, sig, z = model.encode(X)
        pred       = model.decode(z)
        mup, _, _  = model.encode(pred)
        mse = mse_loss(X, pred)
        kl  = kl_loss(mu, sig, mup, sig0=-4)

        loss = mse + kl_beta*kl

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print_loss += loss
        print_mse  += mse
        print_kl   += kl

        if batch % BATCH_PRINT == 0:
            loss = loss.item()

            print_loss /= BATCH_PRINT
            print_mse  /= BATCH_PRINT
            print_kl   /= BATCH_PRINT
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            pbar.set_postfix_str(f'loss: {print_loss:5.2f} mse: {print_mse:5.2f} kl: {print_kl:5.2f}')
            print_loss, print_mse, print_kl = 0, 0, 0

def test_VAE(data, model, kl_beta):
    num_batches = len(data)
    model.eval()
    test_loss, mse, kl = 0, 0, 0
    with torch.no_grad():
        for x in tqdm.tqdm(data):
            X = torch.Tensor(x).to(device)
            mu, sig, z = model.encode(X)
            pred       = model.decode(z)
            mup, _, _  = model.encode(pred)
            mse += mse_loss(X, pred)
            kl  += kl_loss(mu, sig, mup, sig0=-4)
            # test_loss += mse + 0.1*kl
    mse /= num_batches
    kl  /= num_batches
    test_loss = mse + kl_beta*kl
    print(f"Test Avg loss: {test_loss:>8f} mse: {mse:>8f} kl: {kl:>8f} \n")

def get_z(data, model, batch_size=16):
    if isinstance(data, DataGenerator):
        dataloader = data
        ndata      = data.ndata
    else:
        dataloader = NumpyGenerator(data, batch_size=batch_size)
        ndata      = len(data)

    model.eval()

    _, _, ztest = model.encode(torch.Tensor(dataloader[0]).to(device))
    _, zshape   = ztest.shape

    z   = np.zeros((ndata, zshape))
    mu  = np.zeros((ndata, zshape))
    sig = np.zeros((ndata, zshape))

    with torch.no_grad():
        for i, x in enumerate(tqdm.tqdm(dataloader)):
            X = torch.Tensor(x).to(device)
            mui, sigi, zi = model.encode(X)
            z[i*dataloader.batch_size:(i+1)*dataloader.batch_size]   = zi.cpu().numpy()
            mu[i*dataloader.batch_size:(i+1)*dataloader.batch_size]  = mui.cpu().numpy()
            sig[i*dataloader.batch_size:(i+1)*dataloader.batch_size] = sigi.cpu().numpy()

    return mu, sig, z

def get_recon(data, model, batch_size=16):
    if isinstance(data, DataGenerator):
        dataloader = data
        img_shape  = dataloader[0].shape[1:]
    else:
        dataloader = NumpyGenerator(data, batch_size=batch_size)
        img_shape  = data[0].shape

    print(len(dataloader), img_shape)
    model.eval()

    recon = np.zeros((dataloader.ndata, *img_shape))
    print(recon.shape)

    with torch.no_grad():
        for i, x in enumerate(dataloader):
            X = torch.Tensor(x).to(device)
            recon[i*dataloader.batch_size:(i+1)*dataloader.batch_size] = model(X).cpu().numpy()

    return recon

def train_VAE(model, train_data, test_data, kl_beta=0.1, epochs=150, lr=1.e-4):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for t in range(epochs):
        test_data.shuffle(); train_data.shuffle();

        print(f"Epoch {t+1}\n-------------------------------")
        train_VAE_batch(train_data, model, optimizer, kl_beta=kl_beta)
        test_VAE(test_data, model, kl_beta=kl_beta)


class Trainer:
    def __init__(self, model, train_data, test_data, kl_beta=0.1, losses=['mse', 'kl_loss'], metrics=['mse','kl_loss']):
        self.model = model

        self.train_data = train_data
        self.test_data  = test_data

        self.train_batch_size = train_data.batch_size
        self.test_batch_size  = test_data.batch_size

        self.kl_beta = kl_beta

        self.losses  = losses
        self.metrics = dict.fromkeys(metrics)

    def loss(self, X):
        loss = torch.zeros(1).to(device)
        pred = self.model(X)

        for loss_name in self.losses:
            if loss_name=='mse':
                lossi = mse_loss(X, pred)
            elif loss_name=='kl_loss':
                mu, sig, z = self.model.encode(X)
                mup, _, _  = self.model.encode(pred)
                lossi = self.kl_beta*kl_loss(mu, sig, mup, sig0=-4)
            else:
                raise ValueError(f'Loss {loss_name} not implemented')
            if loss_name in self.metrics.keys():
                self.metrics[loss_name] = lossi.cpu()
            loss += lossi

        return loss

    def train_batch(self, optimizer):
        size = len(self.train_data)
        self.model.train()
        pbar = tqdm.tqdm(self.train_data)
        for batch, x in enumerate(pbar):
            X = torch.Tensor(x).to(device)

            loss = self.loss(X)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics = ""
            for key, metric in self.metrics.items():
                metrics += f"{key}: {metric:5.2f}"

            pbar.set_postfix_str(f'loss: {loss.item():5.2f} {metrics}')

    def test(self):
        num_batches = len(self.test_data)
        self.model.eval()
        test_loss = 0
        test_metrics = {}
        for metric in self.metrics:
            test_metrics[metric] = 0

        with torch.no_grad():
            for x in tqdm.tqdm(self.test_data):
                X = torch.Tensor(x).to(device)

                test_loss += self.loss(X).cpu()
                for metric in self.metrics:
                    test_metrics[metric] += self.metrics[metric]

        test_loss /= num_batches

        metrics = ""
        for metric in self.metrics:
            test_metrics[metric] /= num_batches
            metrics += f"{metric}: {test_metrics[metric]:5.2f} "

        print(f"Test Avg loss: {test_loss.item():>8f} {metrics}")

    def train(self, epochs, optim='Adam', lr=1.e-4, val_freq=5, checkpoint_freq=10, savepath='checkpoints/') :
        if optim=='Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"optimizer {optim} not implemented.")

        if not os.path.exists(savepath):
            os.mkdir(savepath)

        for t in range(epochs):
            self.train_data.shuffle();

            print(f"Epoch {t+1}\n-------------------------------")
            self.train_batch(optimizer)

            if (t+1)%val_freq==0:
                self.test_data.shuffle();
                self.test()

            if (t+1)%checkpoint_freq==0:
                checkpoint_path = f"{savepath}checkpoint-{t+1:05d}.pth"
                torch.save(self.model.state_dict(), checkpoint_path) 
                print(f"Saved PyTorch Model State to {checkpoint_path}")

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

        _, _, ztest = self.model.encode(torch.Tensor(data[0]).to(device))
        _, zshape   = ztest.shape

        z   = np.zeros((data.ndata, zshape))
        mu  = np.zeros((data.ndata, zshape))
        sig = np.zeros((data.ndata, zshape))

        with torch.no_grad():
            for i, x in enumerate(tqdm.tqdm(data)):
                X = torch.Tensor(x).to(device)
                mui, sigi, zi = self.model.encode(X)
                z[i*data.batch_size:(i+1)*data.batch_size]   = zi.cpu().numpy()
                mu[i*data.batch_size:(i+1)*data.batch_size]  = mui.cpu().numpy()
                sig[i*data.batch_size:(i+1)*data.batch_size] = sigi.cpu().numpy()

        return mu, sig, z

    def get_recon(self, data_sub=None, batch_size=16):
        if data_sub is None:
            data = self.data
        else:
            if not isinstance(data_sub, DataGenerator):
                data = NumpyGenerator(data_sub, batch_size=batch_size)
            else:
                data = data

        print(len(data), self.img_shape)
        self.model.eval()

        recon = np.zeros((data.ndata, *self.img_shape))
        print(recon.shape)

        with torch.no_grad():
            for i, x in enumerate(data):
                X = torch.Tensor(x).to(device)
                recon[i*data.batch_size:(i+1)*data.batch_size] = self.model(X).cpu().numpy()

        return recon
