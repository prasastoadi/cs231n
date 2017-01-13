import pickle
import numpy as np
import os
from scipy.misc import imread


def dict_bytes_to_strings(dicti):
    """ konversi dari bytes ke strings

    Contoh:
    - dicti_bytes = {b'nama': b'aku'}
    - dicti_strings = dict_bytes_to_strings(dicti)
    - print(dicti_strings)
    - {'nama': 'aku'}
    """
    if isinstance(dicti, bytes): return dicti.decode()
    if isinstance(dicti, tuple): return tuple(map(dict_bytes_to_strings, dicti))
    if isinstance(dicti, list): return list(map(dict_bytes_to_strings, dicti))
    if isinstance(dici, dict): return dict(map(dict_bytes_to_strings, dicti.items()))
    return dicti

def convert_CIFAR_py2_to_py3(filename, newname=None, postfix='3'):
    """ konversi CIFAR 10 Python 2 pickle to Python 3

    Returns:
    Procedure, no returns

    Output:
    CIFAR data in pickle HIGHEST_PROTOCOL
    """
    with open(filename, 'rb') as batch_file:
        batch_data = pickle.load(batch_file, encoding='bytes')
    
    batch_data = dict_bytes_to_strings(batch_data)

    if not newname: newname = filename + postfix

    with open(newname, 'wb+') as batch_file:     
        pickle.dump(batch_data, batch_file, protocol=pickle.HIGHEST_PROTOCOL)
        
def convert_CIFAR(dirname, postfix='3'):
    """convert all batches in directory"""
    newdir = '{0} {1}'.format(dirname, postfix)
    os.makedirs(newdir, exist_ok=True)
    for i in range(1,6):
        filename = os.path.join(dirname, 'data_batch_%d'%(i))
        newname =  os.path.join(newdir, 'data_batch_%d'%(i))
        convert_CIFAR_py2_to_py3(filename, newname, postfix='')

    filename = os.path.join(dirname, 'test_batch')
    newname =  os.path.join(newdir, 'test_batch')
    convert_CIFAR_py2_to_py3(filename, newname, postfix='')

def load_CIFAR_batch(filename):
    """ memuat single batch dari CIFAR 10

    Returns:
    Numpy array dengan dimensi (10000, 32, 32, 3)
    """
    with open(filename, 'rb') as batch_file:
        batch_data = pickle.load(batch_file, encoding='bytes')
        #batch_data = dict_bytes_to_strings(batch_data)
        gambar = batch_data['data']
        label = batch_data['labels']
        gambar = gambar.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        label = np.array(label)
        return gambar, label

def load_CIFAR10(ROOT):
    """ memuat semua dataset CIFAR 10 """
    gambar = []
    label = []
    for i in range(1,6):
        batch_filename = os.path.join(ROOT, 'data_batch_%d'%(i))
        batch_gambar, batch_label = load_CIFAR_batch(batch_filename)
        gambar.append(batch_gambar)
        label.append(batch_label)
    train_gambar = np.concatenate(gambar)
    train_label = np.concatenate(label)
    del gambar, label
    test_gambar, test_label = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return train_gambar, train_label, test_gambar, test_label