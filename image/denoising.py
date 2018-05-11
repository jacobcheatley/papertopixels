import cv2
import numpy as np


def nabla(I):
    h, w = I.shape
    G = np.zeros((h, w, 2), I.dtype)
    G[:, :-1, 0] -= I[:, :-1]
    G[:, :-1, 0] += I[:, 1:]
    G[:-1, :, 1] -= I[:-1]
    G[:-1, :, 1] += I[1:]
    return G


def nablaT(G):
    h, w = G.shape[:2]
    I = np.zeros((h, w), G.dtype)
    # note that we just reversed left and right sides
    # of each line to obtain the transposed operator
    I[:, :-1] -= G[:, :-1, 0]
    I[:, 1:] += G[:, :-1, 0]
    I[:-1] -= G[:-1, :, 1]
    I[1:] += G[:-1, :, 1]
    return I


# little auxiliary routine
def anorm(x):
    """Calculate L2 norm over the last array dimention"""
    return np.sqrt((x * x).sum(-1))


def calc_energy_TVL1(X, observation, clambda):
    Ereg = anorm(nabla(X)).sum()
    Edata = clambda * np.abs(X - observation).sum()
    return Ereg + Edata


def project_nd(P, r):
    """perform a pixel-wise projection onto R-radius balls"""
    nP = np.maximum(1.0, anorm(P) / r)
    return P / nP[..., np.newaxis]


def shrink_1d(X, F, step):
    """pixel-wise scalar srinking"""
    return X + np.clip(F - X, -step, step)


def solve_TVL1(img, clambda, iter_n=101):
    # setting step sizes and other params
    L2 = 8.0
    tau = 0.02
    sigma = 1.0 / (L2 * tau)
    theta = 1.0

    X = img.copy()
    P = nabla(X)
    for i in range(iter_n):
        P = project_nd(P + sigma * nabla(X), 1.0)
        X1 = shrink_1d(X - tau * nablaT(P), img, clambda * tau)
        X = X1 + theta * (X1 - X)

    return X


def denoise(img):
    lambda_TVL1 = 1.0
    img = img / 255.0
    img = solve_TVL1(img, lambda_TVL1, iter_n=30)
    img = (255 * img).astype(np.uint8)
    return img


if __name__ == '__main__':
    import time

    img = cv2.imread('/home/jcheatley/PycharmProjects/papertopixels/image/out/85-1525915984.5482326/scaled.png')
    start = time.time()
    img = denoise(img)
    print(time.time() - start)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
