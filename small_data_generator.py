import zarr
import numpy as np 

seismic_data_filename = "data/F3_train.npy"

def filter_data(data): return data[0:10,:,:]

# Read the seismic data samples from a NPY file
seismic_data   = filter_data(np.load(seismic_data_filename))

zarr.save("data/F3_train_XS.zarr", seismic_data)

