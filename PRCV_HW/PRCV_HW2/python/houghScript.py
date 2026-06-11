import cv2
import numpy as np
from myHoughTransform import myHoughTransform
from myHoughLines import myHoughLines


def image_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def points_for_line(rho, theta, H, W):
    # y = (-cosθ x + ρ) / sin
    if abs(np.sin(theta)) < 1e-5:
        return (int(rho), 0), (int(rho), H - 1)
    else:
        x1, y1 = 0, int(rho / np.sin(theta))
        x2, y2 = int(rho / np.cos(theta)), 0
        x3, y3 = W - 1, int((rho - np.cos(theta) * (W - 1)) / np.sin(theta))
        x4, y4 = int((rho - np.sin(theta) * (H - 1)) / np.cos(theta)), H - 1
        points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        point_1, point_2 = 0, 3
        while points[point_1][0] >= W or points[point_1][0] < 0 or points[point_1][1] >= H or points[point_1][1] < 0:
            point_1 += 1
        while points[point_2][0] >= W or points[point_2][0] < 0 or points[point_2][1] >= H or points[point_2][1] < 0:
            point_2 -= 1
        return points[point_1], points[point_2]


def main():
    img = cv2.imread('../data/img01.jpg')
    image_show('origin_image', img)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_threshold = cv2.Canny(img_gray, 100, 200)
    image_show('img_threshold', img_threshold)

    rhoRes, thetaRes = 2, np.pi / 180
    img_hough, rhoScale, thetaScale = myHoughTransform(img_threshold, rhoRes, thetaRes)
    img_hough = np.clip(img_hough, 0, 255).astype(np.uint8)
    image_show('img_hough', img_hough)

    nLines = 20
    rhos, thetas = myHoughLines(img_hough, nLines)
    rhos = rhos * rhoRes
    thetas = thetas * thetaRes

    H, W = img.shape[:2]
    img_red = img.copy()
    for rho, theta in zip(rhos, thetas):
        point_1, point_2 = points_for_line(rho, theta, H, W)
        cv2.line(img_red, point_1, point_2, (0, 0, 255), 2)
    image_show('img_red', img_red)
    cv2.imwrite('../result/result_self_Hough.png', img_red)

    # 使用 OpenCV 的 HoughLinesP 作为对照（绿色线段）
    img_green = img.copy()
    lines = cv2.HoughLines(img_threshold, rhoRes, thetaRes, threshold=100)
    for line in lines[:min(nLines, len(lines))]:
        rho, theta = line[0]
        point_1, point_2 = points_for_line(rho, theta, H, W)
        cv2.line(img_green, point_1, point_2, (0, 255, 0), 2)
    image_show('img_green', img_green)
    cv2.imwrite('../result/result_cv2_Hough.png', img_green)


if __name__ == '__main__':
    main()