import numpy as np


def myHoughTransform(img_threshold, rhoRes, thetaRes):
    row, col = img_threshold.shape
    max_rho, max_theta = np.sqrt(row ** 2 + col ** 2), 2 * np.pi
    rhoScale = np.arange(0, max_rho, rhoRes)
    thetaScale = np.arange(0, max_theta, thetaRes)
    cos_theta, sin_theta = np.cos(thetaScale), np.sin(thetaScale)
    img_hough = np.zeros((len(rhoScale), len(thetaScale)), dtype=np.uint32)
    ys, xs = np.nonzero(img_threshold)
    for x, y in zip(xs, ys):
        rhos = x * cos_theta + y * sin_theta
        valid_idx = rhos >= 0
        rhos_valid = rhos[valid_idx]
        rhos_indices = np.array(rhos_valid / rhoRes, dtype=int)
        thetas_indices = np.where(valid_idx)[0]
        for i, j in zip(rhos_indices, thetas_indices):
            img_hough[i, j] += 1
    return img_hough, rhoScale, thetaScale
