import h5py

def get_data(file):
    f = h5py.File(file)
    data = f['data'][:]
    label = f['label'][:]
    
    return data, label