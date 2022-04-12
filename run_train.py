from convAE import *
from torchinfo import summary

if __name__=='__main__':
    conv_filt     = 512
    hidden        = [128, 16]
    model = BaseVAE(conv_filt=conv_filt, hidden=hidden).to(device)

    train_data, test_data = create_generators('../junodata/segments_20220408_384.nc', 12)

    #print(model)
    #summary(model, input_size=(32, 3, 384, 384), device=device)

    #train_VAE(model, train_data, test_data, lr=1.e-5)

    trainer = Trainer(model, train_data, test_data)

    trainer.train(epochs=150)

    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")
