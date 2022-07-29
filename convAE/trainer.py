from .model import *
from .losses import *
from torch.optim.lr_scheduler import ExponentialLR
import os
import glob

class Trainer:
    '''
        Object used for model training
    '''
    def __init__(self, model, train_data, test_data, kl_beta=0.1, clust_beta=0.1, losses=['mse', 'kl_loss']):
        self.model = model

        self.train_data = train_data
        self.test_data  = test_data

        self.train_batch_size = train_data.batch_size
        self.test_batch_size  = test_data.batch_size

        self.kl_beta    = kl_beta
        self.clust_beta = clust_beta

        self.losses  = losses
        self.metrics = dict.fromkeys(losses)

        self.start   = 0

    def load_last_checkpoint(self, savepath='checkpoints/'):
        checkpoints = sorted(glob.glob(savepath+"checkpoint-*.pth"))

        assert len(checkpoints) > 0, "No checkpoints found!"

        self.load_from_checkpoint(checkpoints[-1])

    def load_from_checkpoint(self, checkpoint_path):

        assert os.path.exists(checkpoint_path), f"{checkpoint_path} not found!"

        self.model.load_state_dict(torch.load(checkpoint_path)['model'])
        if hasattr(self, 'disc'):
            self.disc.load_state_dict(torch.load(checkpoint_path)['discriminator'])

        self.start = int(checkpoint_path.split('/')[-1].replace('checkpoint-','')[:-4])

        print(f"Loaded model state from {checkpoint_path} for epoch {self.start}")

    def get_loss(self, X):
        self.loss = torch.zeros(1).to(device)
        if 'DEC' not in self.model.type:
            pred = self.model(X)
        else:
            pred, gamma = self.model(X)

        for loss_name in self.losses:
            if loss_name=='mse':
                lossi = mse_loss(X, pred)
            elif loss_name=='kl_loss':
                mu, sig, z = self.model.encode(X)
                mup, _, _  = self.model.encode(pred)
                lossi = self.kl_beta*kl_loss(mu, sig, mup, sig0=-4)
            elif loss_name=='dec_loss':
                lossi = self.clust_beta*dec_loss(gamma)
            else:
                raise ValueError(f'Loss {loss_name} not implemented')
            if loss_name in self.metrics.keys():
                self.metrics[loss_name] = lossi.item()
            self.loss += lossi

        if hasattr(self, 'disc'):
            # add the binary crossentropy loss
            # from the discriminator
            '''
            crops_pred = torch.empty((X.shape[0], 25, 3, 76, 76), dtype=torch.float32).to(device)
            crops_true = torch.empty((X.shape[0], 25, 3, 76, 76), dtype=torch.float32).to(device)

            # crop the images into 5 segments on each side
            # to correspond to what the final latent space vector
            # looks like. so we are going to test the reconstruction
            # one segment at a time
            for j in range(5):
                for i in range(5):
                    crops_pred[:,j*5+i,:,:] = pred[:,:,j*77:j*77+76,i*77:i*77+76]
                    crops_true[:,j*5+i,:,:] = X[:,:,j*77:j*77+76,i*77:i*77+76]

            crops_pred = torch.flatten(crops_pred, start_dim=0, end_dim=1)
            crops_true = torch.flatten(crops_true, start_dim=0, end_dim=1)
            '''
            
            label_pred = self.disc(self.model.decode(torch.randn((X.shape[0], 200), dtype=torch.float).to(device)))
            label_true = self.disc(X)

            disc_loss1 = nn.BCELoss()(label_pred, torch.zeros_like(label_pred))
            disc_loss2 = nn.BCELoss()(label_true, torch.ones_like(label_true))
            self.disc_loss = (disc_loss1 + disc_loss2)*350
            self.metrics['disc_loss'] = self.disc_loss.item()

        if 'DEC' not in self.model.type:
            return pred
        else:
            return pred, gamma

    def train_batch(self):
        '''
            Train a single epoch of the model
        '''
        size = len(self.train_data)
        self.model.train()
        if hasattr(self, 'disc'):
            self.disc.train()
        pbar = tqdm.tqdm(self.train_data)

        losses = []

        for batch, x in enumerate(pbar):
            # account for value and ground truth
            if len(x) == 2:
                x = x[0]
            X = torch.Tensor(x).to(device)


            pred = self.get_loss(X)

            # Backpropagation
            if hasattr(self, 'disc'):
                self.optim_disc.zero_grad()
                self.disc_loss.backward(retain_graph=True)
                self.optim_disc.step()

            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()


            # add the metrics to the loss history array
            losses.append([self.loss.item()])
            for key, metric in self.metrics.items():
                losses[-1].append(metric)

            # for printing out calculate the average over the epoch
            metric_keys  = ['loss', *self.metrics.keys()]
            mean_metrics = np.mean(losses, axis=0)
            metrics = ""
            for i in range(len(metric_keys)):
                metric = mean_metrics[i]
                if metric < 1.e-2:
                    metrics += f"{metric_keys[i]}: {metric:5.2e} "
                else:
                    metrics += f"{metric_keys[i]}: {metric:5.2f} "

            # add gamma min/max for DEC models so we can track clustering performance
            if 'DEC' in self.model.type:
                gamma = pred[1]
                metrics += f"gamma: {gamma.min():3.2e} {gamma.max():3.2e}"
                #metrics += f"clust: {self.model.DEC.cluster_centers.min():5.2e} {self.model.DEC.cluster_centers.max():5.2e}"

            pbar.set_postfix_str(metrics)

        return np.mean(losses, axis=0)

    def test(self):
        '''
            Evaluate the model performance on the validation/test data
        '''
        num_batches = len(self.test_data)
        self.model.eval()

        if hasattr(self, 'disc'):
            self.disc.eval()
        test_loss = 0
        test_metrics = {}
        for metric in self.metrics:
            test_metrics[metric] = 0

        with torch.no_grad():
            for x in tqdm.tqdm(self.test_data):
                # account for value and ground truth
                if len(x) == 2:
                    x = x[0]
                X = torch.Tensor(x).to(device)
                
                self.get_loss(X)

                test_loss += self.loss.cpu()
                for metric in self.metrics:
                    test_metrics[metric] += self.metrics[metric]

        test_loss /= num_batches
        losses = [test_loss.item()]

        metrics = ""
        for metric in self.metrics:
            test_metrics[metric] /= num_batches
            metrics += f"{metric}: {test_metrics[metric]:5.2f} "
            losses.append(test_metrics[metric])

        print(f"Test Avg loss: {test_loss.item():>8f} {metrics}")

        return losses

    def train(self, epochs, optim='Adam', lr=1.e-4, val_freq=5, checkpoint_freq=10, savepath='checkpoints/', lr_decay=None, decay_freq=5):
        '''
            Train the model. Runs through `epochs` number of epochs given the input `optim` optimizer with 
            a learning rate defined by `lr`. Saves checkpoints every `checkpoint_freq` epochs and calculates 
            the loss from the validation sample every `val_freq` epochs.
        '''
        if optim=='Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optim=='SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"optimizer {optim} not implemented.")

        if hasattr(self, 'disc'):
            self.optim_disc = torch.optim.Adam(self.disc.parameters(), lr=1.e-3)

        if lr_decay is not None:
            scheduler = ExponentialLR(self.optimizer, gamma=lr_decay)
        else:
            scheduler = None

        # make the save folder if it doesn't exist
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        train_loss_history = []
        test_loss_history  = []
        
        if os.path.exists(f"{savepath}history.npz"):
            loss_hist = np.load(f"{savepath}history.npz", allow_pickle=True)

            if len(loss_hist['train_loss']) >= self.start:
                train_loss_history = loss_hist['train_loss'].tolist()[:self.start]
                test_loss_history  = loss_hist['test_loss']
                tmask = [thist[0] <= self.start for thist in test_loss_history]
                test_loss_history  = test_loss_history[tmask].tolist()
                print(f"Loaded history from {savepath}history.npz")

        for t in range(self.start, epochs):
            if hasattr(self.train_data, 'shuffle'):
                self.train_data.shuffle();

            if scheduler is not None:
                lr = scheduler.get_last_lr()[0]

            print(f"Epoch {t+1} -- Learning rate: {lr:5.3e}\n----------------------------------")
            train_loss = self.train_batch()
            train_loss_history.append([t, *train_loss])

            if (t+1)%val_freq==0:
                if hasattr(self.test_data, 'shuffle'):
                    self.test_data.shuffle();
                test_loss = self.test()
                test_loss_history.append([t, *test_loss])

            if (t+1)%checkpoint_freq==0:
                self.save_checkpoint(t+1, savepath)
                np.savez(f"{savepath}history.npz", train_loss=train_loss_history, test_loss=test_loss_history)
            
            if scheduler is not None:
                if (t+1)%decay_freq==0:
                    lr = scheduler.step()
        
        torch.save(self.model.state_dict(), f"{savepath}model.pth")
        print(f"Saved PyTorch Model State to {savepath}model.pth")
        np.savez(f"{savepath}history.npz", train_loss=train_loss_history, test_loss=test_loss_history)

    def save_checkpoint(self, t, savepath):
        checkpoint_path = f"{savepath}checkpoint-{t:05d}.pth"

        if hasattr(self, 'disc'):
            torch.save({'model': self.model.state_dict(), 'discriminator': self.disc.state_dict()}, 
                   checkpoint_path) 
        else:
            torch.save({'model': self.model.state_dict()}, checkpoint_path) 

        print(f"Saved PyTorch Model State to {checkpoint_path}")
