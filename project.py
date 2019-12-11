#####################################################################################
# original template code
import cv2
import numpy as np

group = "group1"


image = cv2.imread("images/{}.jpg".format(group))

cv2.imshow("image", image)
cv2.waitKey(0)

#####################################################################################

# convert to gray-scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# CLAHE contrast enhancement
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
image_clahe = clahe.apply(gray)

# black hair removal
rectKernelBlack = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
blackhat = cv2.morphologyEx(image_clahe, cv2.MORPH_BLACKHAT, rectKernelBlack)
(T, threshBlack) = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
inpaintedBlack = cv2.inpaint(image, threshBlack, 3, cv2.INPAINT_TELEA)

cv2.imshow("noHair", inpaintedBlack)
cv2.waitKey(0)
cv2.imwrite("out/{}/hairRemoval.jpg".format(group), inpaintedBlack)