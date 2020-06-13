import numpy as np
import cv2 as cv
import imutils
from scipy import ndimage
from skimage.measure import compare_ssim
import collections


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
    # kernel7 = np.ones((5, 1), np.uint8)
    kernel3 = np.ones((7, 7), np.uint8)
    kernel4 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((9, 9), np.uint8)
    # kernel6 = np.ones((5, 5), np.uint8)
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


def save_plate(image, image_in):
    possible_plates = {}
    config = {
        "w_min": 100,
        "w_max": 640,
        "h_min": 50,
        "h_max": 480,
        "w_ref": 520,
        "h_ref": 100,
    }
    ref_ar = int(config["w_ref"] / config["h_ref"])
    prox = 5
    cnts = cv.findContours(image.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:30]
    #
    for c in cnts:
        rect = cv.minAreaRect(c)
        _, (w, h), _ = rect
        if int(w / h) - prox <= ref_ar <= int(w / h) + prox and config["w_min"] < w and config["h_min"] < h:
            possible_plates[int(w * h)] = c
    max_sized_plate = max(possible_plates.keys())
    plate_cnt = possible_plates[max_sized_plate]
    plate = cv.minAreaRect(plate_cnt)
    box = cv.boxPoints(plate)
    box = np.int0(box)
    width = int(plate[1][0])
    height = int(plate[1][1])
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]], dtype="float32")

    M = cv.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv.warpPerspective(image_in, M, (width, height))
    if warped.shape[0] > warped.shape[1]:
        warped = cv.rotate(warped, cv.ROTATE_90_COUNTERCLOCKWISE)
    cv.drawContours(image_in, [box], 0, (0, 0, 255), 2)
    return warped

# TODO: improve character detection
def process_plate(image):
    locs = []
    significant = []


    kernel = np.ones((5, 5), dtype=np.uint8)
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_gray = cv.bilateralFilter(image_gray, 11, 17, 17)
    image_gray = cv.GaussianBlur(image_gray, (5, 5), 0)
    image_eq = cv.equalizeHist(image_gray)
    image_thresh = cv.adaptiveThreshold(image_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv.THRESH_BINARY, 15, 2)
    cnts, h = cv.findContours(image_thresh.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE, hierarchy=True)
    # cnts = imutils.grab_contours(cnts)
    # cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:30]
    for i, tupl in enumerate(h[0]):
        if tupl[3] != -1:
            tupl = np.insert(tupl, 0, [i])
            locs.append(tupl)

    for tupl in locs:
        contour = cnts[tupl[0]]
        (x, y, w, h) = cv.boundingRect(contour)
        ar = float(h) / float(w)
        if ar > 0.5 and ar < 4:
            if h > 40 and h < 150 and w > 10 and w < 80:
                significant.append((x, y, w, h))

    for element in significant:
        (x, y, w, h) = element
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow('Plate', image)
    cv.imshow('Plate_Canny', image_thresh)
    cv.waitKey(2000)


def perform_processing(image: np.ndarray) -> str:
    img = image
    img = cv.resize(img, (640, 480))
    img_thresh = extract_plate(img)
    plate = save_plate(img_thresh, img)
    process_plate(plate)
    return 'XXXXXXX'
