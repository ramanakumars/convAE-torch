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

        self.model.load_state_dict(torch.load(checkpoints[-1]))

        self.start = int(checkpoints[-1].split('/')[-1].replace('checkpoint-','')[:-4])
        print(f"Loaded model state from {checkpoints[-1]} for epoch {self.start}")

    def load_from_checkpoint(self, checkpoint_path):

        assert os.path.exists(checkpoint_path), f"{checkpoint_path} not found!"

        self.model.load_state_dict(torch.load(checkpoint_path))

        self.start = int(checkpoint_path.split('/')[-1].replace('checkpoint-','')[:-4])

        print(f"Loaded model state from {checkpoint_path} for epoch {self.start}")

    def loss(self, X):
        loss = torch.zeros(1).to(device)
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
            loss += lossi

        if 'DEC' not in self.model.type:
            return loss, pred
        else:
            return loss, (pred, gamma)

    def train_batch(self, optimizer):
        '''
            Train a single epoch of the model
        '''
        size = len(self.train_data)
        self.model.train()
        pbar = tqdm.tqdm(self.train_data)

        losses = []

        for batch, x in enumerate(pbar):
            # account for value and ground truth
            if len(x) == 2:
                x = x[0]
            X = torch.Tensor(x).to(device)


            loss, pred = self.loss(X)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # add the metrics to the loss history array
            losses.append([loss.item()])
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

                test_loss += self.loss(X)[0].cpu()
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
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"optimizer {optim} not implemented.")

        if lr_decay is not None:
            scheduler = ExponentialLR(optimizer, gamma=lr_decay)
        else:
            scheduler = None

        # make the save folder if it doesn't exist
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        train_loss_history = []
        test_loss_history  = []
        
        if os.path.exists(f"{savepath}history.npz"):
            loss_hist = np.load(f"{savepath}history.npz")

            if len(loss_hist['train_loss']) >= self.start:
                train_loss_history = loss_hist['train_loss'].tolist()[:self.start]
                test_loss_history  = loss_hist['test_loss']
                test_loss_history  = test_loss_history[test_lost_history[:,0]<=self.start].tolist()
                print(f"Loaded history from {savepath}history.npz")

        for t in range(self.start, epochs):
            if hasattr(self.train_data, 'shuffle'):
                self.train_data.shuffle();

            if scheduler is not None:
                lr = scheduler.get_last_lr()[0]

            print(f"Epoch {t+1} -- Learning rate: {lr:5.3e}\n----------------------------------")
            train_loss = self.train_batch(optimizer)
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
        torch.save(self.model.state_dict(), checkpoint_path) 
        print(f"Saved PyTorch Model State to {checkpoint_path}")
