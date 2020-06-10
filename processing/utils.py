import numpy as np
import cv2 as cv
import imutils
import math
from scipy import ndimage
from skimage.measure import compare_ssim

# Roberts edge detection algorithm


def roberts_cross(image):
    roberts_cross_v = np.array([[0, 0, 0],
                                [0, 1, 0],
                                [0, 0, -1]])

    roberts_cross_h = np.array([[0, 0, 0],
                                [0, 0, 1],
                                [0, -1, 0]])

    image = image.astype(np.int32)
    vertical = ndimage.convolve(image, roberts_cross_v)
    horizontal = ndimage.convolve(image, roberts_cross_h)
    output_image = np.sqrt(np.square(horizontal) + np.square(vertical))
    return output_image.astype(np.uint8)


def extract_plate(image):
    kernel1 = np.ones((1, 7), np.uint8)
    kernel2 = np.ones((7, 1), np.uint8)
    kernel3 = np.ones((7, 7), np.uint8)
    kernel4 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((9, 9), np.uint8)
    kernel6 = np.ones((5, 5), np.uint8)

    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_gray = cv.bilateralFilter(image_gray, 5, 75, 75)
    image_roberts = roberts_cross(image_gray)
    image_roberts = cv.dilate(image_roberts, kernel4, iterations=1)
    # image_roberts = cv.morphologyEx(image_roberts, cv.MORPH_CLOSE, kernel6, iterations=1)
    image_opened = cv.morphologyEx(image_roberts, cv.MORPH_OPEN, kernel1, iterations=1)
    image_closed = cv.morphologyEx(image_roberts, cv.MORPH_CLOSE, kernel1, iterations=1)
    (score, image_diff) = compare_ssim(image_opened, image_closed, full=True)
    image_diff = (image_diff * 255).astype("uint8")
    image_diff = cv.morphologyEx(image_diff, cv.MORPH_CLOSE, kernel2, iterations=3)
    image_thresh = cv.threshold(image_diff, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    image_thresh = cv.dilate(image_thresh, kernel6, iterations=8)
    # image_thresh = cv.morphologyEx(image_thresh, cv.MORPH_OPEN, kernel3, iterations=1)

    cv.imshow('original', image)
    cv.imshow('thresh', image_thresh)

    cv.waitKey(5000)


def perform_processing(image: np.ndarray) -> str:
    img = image
    img = cv.resize(img, (640, 480))
    extract_plate(img)
    return 'XXXXXXX'
