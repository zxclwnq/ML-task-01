import numpy as np


def prod_non_zero_diag(x):
    return np.prod(np.diag(x),  where=np.diag(x)!=0)


def are_multisets_equal(x, y):
    return np.array_equal(np.sort(x, kind='heapsort'), np.sort(y, kind='heapsort'))


def max_after_zero(x):
    return np.max(x[1:], where=x[:-1]==0, initial = int(-1e9))

def convert_image(img, coefs):
    gray_v = img.dot(coefs)
    gray_img = gray_v.astype(np.uint8)
    return gray_img


def run_length_encoding(x):
    n = len(x)
    ia = x[1:] != x[:-1]
    i = np.append(np.where(ia), n-1)
    return (x[i], np.diff(np.append(-1, i)))


def pairwise_distance(x, y):
    return np.linalg.norm(x[:, None, :] - y[None, :, :], axis = -1)
