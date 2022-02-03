import cv2
import imutils
import numpy as np
import pytesseract
from PIL import Image

# read image grayscale and blur
img = cv2.imread('image6.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# apply filter and localization
bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
edged = cv2.Canny(bfilter, 30, 200)  # Edge detection

# find contours and apply mask
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break
# loop over our contours
for contour in contours:
    # approximate the contour
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.018 * peri, True)

    # if our approximated contour has four points, then
    # we can assume that we have found our screen
    if len(approx) == 4:
        location = approx
        break

if location is None:
    detected = 0
    print("No contour detected")
else:
    detected = 1

if detected == 1:
    cv2.drawContours(img, [location], -1, (0, 255, 0), 3)

mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0, 255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)

# Now crop
(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]

# Read the number plate
# text = pytesseract.image_to_string(Cropped, config='--psm 11')
# print("Detected Number is:", text)

cv2.imshow('image', img)
cv2.imshow('Cropped', Cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()