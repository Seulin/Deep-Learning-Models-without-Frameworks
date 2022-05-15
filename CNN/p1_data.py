# Import local mnist data
import gzip
import numpy as np

def import_img(num, path):
    f = gzip.open(path, 'r')

    image_size = 28

    f.read(16)
    buf = f.read(image_size * image_size * num)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float64)
    data = data.reshape(num, image_size, image_size)
    f.close()
    return data

def import_lbl(num, path):
    f = gzip.open(path, 'r')

    image_size = 784
    f.read(8)
    
    labels = []
    for i in range(num):
        buf = f.read(1)
        label = np.frombuffer(buf, dtype=np.uint8).astype(np.int32)
        labels.extend(label)
    f.close()
    return np.array(labels)

def data_loader(x_data, y_data, batch_size = 64):
    assert (x_data.shape[0] == y_data.shape[0])

    batch_len, remainder = divmod(x_data.shape[0], batch_size)

    remainder = 1 if remainder != 0 else 0
    for i in range(batch_len + remainder):
        start = i*batch_size
        if i == batch_len + remainder - 1:
            yield x_data[start:], y_data[start:]
        else:
            yield x_data[start:start + batch_size], y_data[start:start + batch_size]
