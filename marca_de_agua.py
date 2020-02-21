import numpy as np
import cv2

from utils import image_resize

img = cv2.imread("saved-media/DeIncognito.PNG", -1)

logo = cv2.imread('misc/logo/logo_ghm2.png', -1)
watermark = image_resize(logo, height=100)

img2 = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
watermark2 = cv2.cvtColor(watermark, cv2.COLOR_BGR2BGRA)
img2_h, img2_w, img2_c = img2.shape
watermark_h, watermark_w, watermark_c = watermark.shape

overlay = np.zeros((img2_h, img2_w, 4), dtype='uint8')

for i in range(0, watermark_h):
        for j in range(0, watermark_w):
            if np.all(watermark[i,j] == 255): # draws logo white pixels only
                offset = 10
                #h_offset = img2_h - watermark_h - offset
                h_offset = offset
                w_offset = img2_w - watermark_w - offset
                overlay[h_offset + i, w_offset+ j] = watermark[i,j]

cv2.addWeighted(overlay, 1, img2, 1.0, 0, img2)

img2 = cv2.cvtColor(img2, cv2.COLOR_BGRA2BGR)

cv2.imwrite("DeIncognito_MA.PNG", img2)