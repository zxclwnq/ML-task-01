import numpy as np

def prod_non_zero_diag(x):
    res = 1
    for i in range(min(len(x), len(x[0]))):
        if res != 0:
            res *= x[i][i]
    return res


def are_multisets_equal(x, y):
    return sorted(x) == sorted(y)


def max_after_zero(x):
    mx = -1e9
    for i in range(1, len(x)):
        if x[i-1] == 0:
            mx = max(mx, x[i])
    return mx


def convert_image(img, coefs):
    k = [[0] * img.shape[1] for _ in range(img.shape[0])]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for z in range(img.shape[2]):
                k[i][j] += img[i][j][z] * coefs[z]
    return np.array(k).astype(np.uint8)
    


def run_length_encoding(x):
    a = [x[0]]
    b = []
    last = 0
    for i in range(1, len(x)):
        if x[i] != x[i-1]:
            b.append(i-last)
            last = i
            a.append(x[i])
    b.append(len(x)-last)
    return (a, b)


def pairwise_distance(x, y):
    ans = [[0] * len(x) for _ in range(len(x))]
    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(x[0])):
                ans[i][j] += (x[i][k] - y[j][k]) ** 2
            ans[i][j] **= 0.5
    return ans
