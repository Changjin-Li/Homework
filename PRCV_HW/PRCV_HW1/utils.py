import sys
import os
import cv2
import numpy as np

def norm_0_255(src):
    if src is None:
        return None
    src = np.float64(src)
    norm_img = cv2.normalize(src, None, 0, 255, cv2.NORM_MINMAX)
    if len(norm_img.shape) == 2:
        norm_img = np.uint8(norm_img)
    else:
        norm_img = np.uint8(norm_img)
    return norm_img

def read_csv(filename, separator=';'):
    images = []
    labels = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(separator)
            if len(parts) < 2:
                continue
            path, class_label = parts[0], parts[1]
            if not path or not class_label:
                continue
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: cannot read images in {path}, pass")
                continue
            images.append(img)
            labels.append(int(class_label))
    return images, labels

def subspaceProject(W, mean, X):
    """
    X: [1, HW]
    mean: [1, HW]
    W: [HW, N]
    Y = W (X - mean): [1, N]
    """
    X = np.asarray(X, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    if W.ndim == 1:
        W = W.reshape(-1, 1)
    return (X - mean) @ W

def subspaceReconstruct(W, mean, Y):
    """X_recon = mean + W Y"""
    Y = np.asarray(Y, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if W.ndim == 1:
        W = W.reshape(-1, 1)
    recon = mean + Y @ W.T
    if recon.shape[0] == 1:
        recon = recon.flatten()
    return recon