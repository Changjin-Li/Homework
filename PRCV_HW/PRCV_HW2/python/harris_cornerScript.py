import cv2
import numpy as np
from scipy import ndimage


def image_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def multi_scale_harris(img, scale_list, window_size, k):
    img = img.astype(np.float32) / 255
    corner_points = []
    responses = []
    for scale in scale_list:
        Ix = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        Iy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        Ix = Ix - np.mean(Ix)
        Iy = Iy - np.mean(Iy)
        Ixx, Ixy, Iyy = Ix * Ix, Ix * Iy, Iy * Iy
        win_size = int(window_size * scale)
        if win_size % 2 == 0:
            win_size = win_size + 1
        G = cv2.getGaussianKernel(win_size, scale)
        Kernel = G @ G.T
        Ixx_s = cv2.filter2D(Ixx, -1, Kernel)
        Iyy_s = cv2.filter2D(Iyy, -1, Kernel)
        Ixy_s = cv2.filter2D(Ixy, -1, Kernel)
        det = Ixx_s * Iyy_s -Ixy_s * Ixy_s
        trace = Ixx_s + Iyy_s
        R = det - k * trace * trace
        responses.append(R)
    responses = np.stack(responses, axis=-1)    # HxWxS
    # 非极大值抑制
    threshold = 0.01 * np.max(responses)
    dilate_window_size = 5
    kernel = np.ones((dilate_window_size, dilate_window_size, dilate_window_size), dtype=bool)
    dilate = ndimage.grey_dilation(responses, footprint=kernel)
    local_max = (responses == dilate) & (responses >= threshold)
    xs, ys, zs = np.where(local_max)
    for x, y, s in zip(xs, ys, zs):
        corner_points.append((x, y, s))
    return corner_points


def main():
    img = cv2.imread('../data/img08.jpg')
    image_show('origin_image', img)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    scale_list = [1.0, 1.5, 2.0, 3.0, 3.5, 4.0, 5.0]
    window_size = 3
    k = 0.1
    corner_points = multi_scale_harris(img_gray, scale_list, window_size, k)

    img_harris = img.copy()
    for point in corner_points:
        cv2.circle(img_harris, (point[1], point[0]), int(scale_list[int(point[2])] * 2), (0, 0, 255), 2)
    image_show('harris_corner', img_harris)
    cv2.imwrite('../result/result_harris_corner.png', img_harris)


if __name__ == "__main__":
    main()