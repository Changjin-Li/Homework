import cv2
import numpy as np


def myHoughLines(img_hough, nLines):
    window_size = 10
    threshold = np.ones_like(img_hough, dtype=np.uint32) * 100
    kernel = np.ones((window_size, window_size), dtype=np.uint8)
    dilate = cv2.dilate(img_hough, kernel)
    local_max = (img_hough == dilate) & (img_hough >= threshold)
    xs, ys = np.where(local_max)
    votes = img_hough[xs, ys]
    sorted_idx = np.argsort(votes)[::-1]
    x_sorted, y_sorted = xs[sorted_idx], ys[sorted_idx]
    n = min(len(sorted_idx), nLines)
    rhos = x_sorted[:n]
    thetas = y_sorted[:n]
    return rhos, thetas
