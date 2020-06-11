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
    kernel7 = np.ones((5, 1), np.uint8)
    kernel3 = np.ones((7, 7), np.uint8)
    kernel4 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((9, 9), np.uint8)
    kernel6 = np.ones((5, 5), np.uint8)
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_gray = cv.bilateralFilter(image_gray, 5, 75, 75)
    image_roberts = roberts_cross(image_gray)
    image_roberts = cv.dilate(image_roberts, kernel4, iterations=1)
    image_opened = cv.morphologyEx(image_roberts, cv.MORPH_OPEN, kernel1, iterations=1)
    image_closed = cv.morphologyEx(image_roberts, cv.MORPH_CLOSE, kernel1, iterations=1)
    (score, image_diff) = compare_ssim(image_closed, image_opened, full=True)
    image_diff = (image_diff * 255).astype("uint8")
    image_diff = cv.morphologyEx(image_diff, cv.MORPH_CLOSE, kernel2, iterations=2)
    image_thresh = cv.threshold(image_diff, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    image_thresh = cv.morphologyEx(image_thresh, cv.MORPH_OPEN, kernel3, iterations=1)
    image_thresh = cv.dilate(image_thresh, kernel5, iterations=7)
    image_thresh = cv.dilate(image_thresh, kernel1, iterations=2)



    return image_thresh

def find_contours(image, image_in):
    correctly_sized = []
    possible_plates = {}
    min_size = 640*480
    config = {
        "w_min": 100,  # char pixel width min
        "w_max": 640,  # char pixel width max
        "h_min": 50,  # char pixel height min
        "h_max": 480,  # char pixel height max
        "hw_min": 1,  # height to width ration min
        "hw_max": 3.5,  # height to width ration max
        "y_offset": 70,  # maximum y offset between chars
        "x_offset": 150,  # maximum x gap between chars
        "h_ave_diff": 1.09,  # acceptable limit for variation between characters
        "x_inbound": 15,
        "w_ref": 520,
        "h_ref": 100,
    }
    ref_ar = int(config["w_ref"]/config["h_ref"])
    prox = 5
    cnts = cv.findContours(image.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:30]
    #
    for c in cnts:
        rect = cv.minAreaRect(c)
        _, (w, h), _ = rect
        if int(w/h) - prox <= ref_ar <= int(w/h) + prox and config["w_min"] < w and config["h_min"] < h:
            possible_plates[int(w*h)] = c
    print(possible_plates)
    max_sized_plate = max(possible_plates.keys())
    plate_cnt = possible_plates[max_sized_plate]
    plate = cv.minAreaRect(plate_cnt)
    box = cv.boxPoints(plate)
    box = np.int0(box)
    cv.drawContours(image_in, [box], 0, (0, 0, 255), 2)
    cv.imshow("Thresholded", image)
    cv.imshow("Contours", image_in)
    cv.waitKey(2000)






def perform_processing(image: np.ndarray) -> str:
    img = image
    img = cv.resize(img, (640, 480))
    img_thresh = extract_plate(img)
    find_contours(img_thresh, img)
    return 'XXXXXXX'
