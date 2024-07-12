import cv2
import numpy as np

for i in range(2, 200):
    anno = np.ones((640, 640, 1), np.uint8)
    for x in range(640):
        for y in range(640):
            anno[y][x] = np.random.randint(1, i)

    cv2.imwrite("gray.png", anno)
    mask = cv2.cvtColor(cv2.imread("gray.png"), cv2.COLOR_BGR2GRAY)
    png_mask_max = mask.max()

    cv2.imwrite("gray.jpg", anno)
    mask = cv2.cvtColor(cv2.imread("gray.jpg"), cv2.COLOR_BGR2GRAY)
    jpg_mask_max = mask.max()

    print(png_mask_max, jpg_mask_max)