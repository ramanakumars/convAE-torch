from .model import *

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

        if batch % 5 == 0:
            loss = loss.item()
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            pbar.set_postfix_str(f'loss: {loss:5.2f} mse: {mse:5.2f} kl: {kl:5.2f}')

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

def train_VAE(model, train_data, test_data, kl_beta=0.1, epochs=150, lr=1.e-4):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_VAE_batch(train_data, model, optimizer, kl_beta=kl_beta)
        test_VAE(test_data, model, kl_beta=kl_beta)
