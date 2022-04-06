from convAE import *
from torchinfo import summary

if __name__=='__main__':
    conv_filt     = 128
    hidden        = [128, 16]
    model = BaseVAE(conv_filt=conv_filt, hidden=hidden).to(device)

    train_data, test_data = create_generators('../junodata/segments_20220214_384.nc', 16)

    #print(model)
    #summary(model, input_size=(32, 3, 384, 384), device=device)

    train_VAE(model, train_data, test_data)

    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")
