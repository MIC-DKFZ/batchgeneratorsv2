import torch
import numpy as np
from time import time


if __name__ == '__main__':
    torch.set_num_threads(1)

    shape = (2, 4, 128, 128, 128)
    print('gaussian noise')
    st = time()
    a = np.random.normal(0, 1, size=shape)
    print(f'numpy: {time() - st}')

    st = time()
    b = torch.from_numpy(a)
    b = torch.normal(0, 1, size=shape)
    b = b.numpy()
    print(f'torch: {time() - st}')
