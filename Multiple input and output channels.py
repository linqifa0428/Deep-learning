import torch
from d2l import torch as d2l


def corr2d_multi_in(X, K):
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))


X=torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
K=torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])

print(corr2d_multi_in(X, K))


def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)


K=torch.stack((K, K+1, K+2), 0)
print(K.shape)

print(corr2d_multi_in_out(X,K))


def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    Y=torch.matmul(K, X)
    return Y.reshape((c_o, h, w))

X=torch.normal(0, 1, (3, 3, 3))
K=torch.normal(0, 1, (2, 3, 1, 1))

Y1=corr2d_multi_in_out_1x1(X, K)
Y2=corr2d_multi_in_out(X, K)
assert float(torch.abs(Y1-Y2).sum())<1e-6