import zarr
import numpy as np 

def filter_data(data): return data[0:10,:,:]

def small_real_data():
    seismic_data_filename = "data/F3_train.npy"
    # Read the seismic data samples from a NPY file
    seismic_data   = filter_data(np.load(seismic_data_filename))
    return seismic_data

# data = small_real_data
# print(np.size(data))
# zarr.save("data/F3_train_XS.zarr", data)

data = np.arange(10*701*255).reshape(10,701,255)
print(np.size(data))
zarr.save("data/F3_fake.zarr", data)




    # def _lazy_transform_cpu(self, data):
    #     f_size = data.size
    #     print(f_size)
    #     features = da.reshape(data.copy(),(1, f_size))

    #     # Get elements for sample window
    #     # Get top neighbor
    #     n1 = data.copy()
    #     # Get bot neighbor
    #     n2 = data.copy()
    #     for i in range(self.samples_window):
    #         # Padding for bottom data
    #         n1[:,:,254] = 0
    #         # Rolling array by 1 position for top neighbor
    #         n1 = da.roll(data, 1, axis=2)
    #         # Adding neighbors in features array
    #         features = da.append(features, da.reshape(n1, (1, f_size)), axis=0)

    #         # Padding for bottom data
    #         n2[:,:,0] = 0
    #         # Rolling array by 1 position for bot neighbor
    #         n2 = da.roll(data, -1, axis=2)
    #         # Adding nArray.a)
    #     # Get bot neighbor
    #     n2 = data.copy()
    #     for i in range(self.trace_window):
    #         # Padding for bottom data
    #         n1[:,700,:] = 0
    #         # Rolling array by 1 position for top neighbor
    #         n1 = da.roll(data, 1, axis=1)
    #         # Adding neighbors in features array
    #         features = da.append(features, da.reshape(n1, (1, f_size)), axis=0)

    #         # Padding for bottom data
    #         n2[:,0,:] = 0
    #         # Rolling array by 1 position for bot neighbor
    #         n2 = da.roll(data, -1, axis=1)
    #         # Adding neighbors in features array
    #         features = da.append(features, da.reshape(n2, (1, f_size)), axis=0)

    #     # Get elements for inline window
    #     # Get top neighbor
    #     n1 = data.copy()
    #     # Get bot neighbor
    #     n2 = data.copy()
    #     for i in range(self.inline_window):
    #         # Padding for bottom data
    #         n1[9,:,:] = 0
    #         # n1[400,:,:] = 0
    #         # Rolling array by 1 position for top neighbor
    #         n1 = da.roll(data, 1, axis=0)
    #         # Adding neighbors in features array
    #         features = da.append(features, da.reshape(n1, (1, f_size)), axis=0)

    #         # Padding for bottom data
    #         n2[0,:,:] = 0
    #         # Rolling array by 1 position for bot neighbor
    #         n2 = da.roll(data, -1, axis=0)
    #         # Adding neighbors in features array
    #         features = da.append(features, da.reshape(n2, (1, f_size)), axis=0)

    #     return features
    
    # def _transform_cpu(self, data):
    #     f_size = data.size(data)
    #     features = np.copy(data).reshape((1, f_size))

    #     # Get elements for sample window
    #     # Get top neighbor
    #     n1 = np.copy(self.ata)
    #     # Get bot neighbor
    #     n2 = np.copy(data)
    #     for i in range(self.samples_window):
    #         # Padding for bottom data
    #         n1[:,:,254] = 0
    #         # Rolling array by 1 position for top neighbor
    #         n1 = np.roll(data, 1, axis=2)
    #         # Adding neighbors in features array
    #         features = np.append(features, np.reshape(n1, (1, f_size)), axis=0)

    #         # Padding for bottom data
    #         n2[:,:,0] = 0
    #         # Rolling array by 1 position for bot neighbor
    #         n2 = np.roll(data, -1, axis=2)
    #         # Adding neighbors in features array
    #         features = np.append(features, np.reshape(n2, (1, f_size)), axis=0)

    #     # Get elements for trace window
    #     # Get top neighbor
    #     n1 = np.copy(data)
    #     # Get bot neighbor
    #     n2 = np.copy(data)
    #     for i in range(self.trace_window):
    #         # Padding for bottom data
    #         n1[:,700,:] = 0
    #         # Rolling array by 1 position for top neighbor
    #         n1 = np.roll(data, 1, axis=1)
    #         # Adding neighbors in features array
    #         features = np.append(features, np.reshape(n1, (1, f_size)), axis=0)

    #         # Padding for bottom data
    #         n2[:,0,:] = 0
    #         # Rolling array by 1 position for bot neighbor
    #         n2 = np.roll(data, -1, axis=1)
    #         # Adding neighbors in features array
    #         features = np.append(features, np.reshape(n2, (1, f_size)), axis=0)

    #     # Get elements for inline window
    #     # Get top neighbor
    #     n1 = np.copy(data)
    #     # Get bot neighbor
    #     n2 = np.copy(data)
    #     for i in range(self.inline_window):
    #         # Padding for bottom data
    #         n1[400,:,:] = 0
    #         # Rolling array by 1 position for top neighbor
    #         n1 = np.roll(data, 1, axis=0)
    #         # Adding neighbors in features array
    #         features = np.append(features, np.reshape(n1, (1, f_size)), axis=0)

    #         # Padding for bottom data
    #         n2[0,:,:] = 0
    #         # Rolling array by 1 position for bot neighbor
    #         n2 = np.roll(data, -1, axis=0)
    #         # Adding neighbors in features array
    #         features = np.append(features, np.reshape(n2, (1, f_size)), axis=0)

    #     return features

