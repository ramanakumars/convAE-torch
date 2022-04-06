import numpy as np
import netCDF4 as nc

class DataGenerator():
    def __init__(self, nc_file, batch_size, indices=None, shuffle=True):
        self.nc_file = nc_file

        self.batch_size = batch_size
        self.shuffle    = shuffle

        if indices is not None:
            self.indices = indices
            self.ndata   = len(indices)
        else:
            with nc.Dataset(nc_file, 'r') as dset:
                self.ndata = int(dset.dimensions['segments'].size)
            self.indices = np.arange(self.ndata)

        print(f"Found data with {self.ndata} images")

    def shuffle(self):
        np.random.shuffle(self.indices)

    def __len__(self):
        return self.ndata // self.batch_size

    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        with nc.Dataset(self.nc_file, 'r') as dset:
            imgs = dset.variables['imgs'][batch_indices,:,:,:]
        imgs[np.isnan(imgs)] = 0.

        return np.transpose(imgs, (0,3,1,2))

def create_generators(nc_file, batch_size, val_split=0.1):
    with nc.Dataset(nc_file, 'r') as dset:
        ndata = dset.dimensions['segments'].size

    print(f"Loading data with {ndata} images")

    inds = np.arange(ndata)
    np.random.shuffle(inds)

    val_split_ind = int(ndata*val_split)
    val_ind      = inds[:val_split_ind]
    training_ind = inds[val_split_ind:]
    
    train_data = DataGenerator(nc_file, batch_size, indices=training_ind)
    val_data   = DataGenerator(nc_file, batch_size, indices=val_ind)

    return train_data, val_data
