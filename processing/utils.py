import numpy as np
import cv2 as cv
import imutils
import math
def extract_plate(image):
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    config = {
        "w_min": 10,  # char pixel width min
        "w_max": 100,  # char pixel width max
        "h_min": 35,  # char pixel height min
        "h_max": 180,  # char pixel height max
        "hw_min": 1,  # height to width ration min
        "hw_max": 3.5,  # height to width ration max
        "y_offset": 70,  # maximum y offset between chars
        "x_offset": 150,  # maximum x gap between chars
        "h_ave_diff": 1.09,  # acceptable limit for variation between characters
        "x_inbound": 15
    }
    correctly_sized = []
    kernel = np.ones((3, 3), np.uint8)
    image = cv.GaussianBlur(image, (5, 5), 0)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)
    gray = cv.bilateralFilter(gray, 11, 17, 17)
    # thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 25, 0)
    thresh = cv.Canny(gray, 10, 200)

    # ret, thresh = cv.threshold(thresh, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # thresh = cv.adaptiveThreshold(thresh, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 25, 0)
    # thresh = cv.morphologyEx(thresh, cv.MORPH_GRADIENT, kernel, iterations=1)
    # ret, thresh = cv.threshold(thresh, 0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    # thresh = cv.adaptiveThreshold(thresh, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 25, 0)
    # thresh = cv.morphologyEx(thresh, cv.MORPH_GRADIENT, kernel, iterations=1)
    # thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=1)
    # thresh = cv.erode(thresh, kernel)

    cnts = cv.findContours(thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:30]
    for c in cnts:
        (x, y, w, h) = cv.boundingRect(c)
        if config["w_min"] < w < config["w_max"] and config["h_min"] < h < config["h_max"]:
            correctly_sized.append((x, y, w, h))
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    correctly_sized = sorted(correctly_sized, key=lambda x: x[0])
    legit = []
    y_values = [i[1] for i in correctly_sized]
    mean_value = np.mean(y_values)
    print(mean_value)
    for i, element in enumerate(correctly_sized):
        if mean_value - config['y_offset'] < element[1] < mean_value + config['y_offset']:
            legit.append(element)

    for element in legit:
        for element1 in legit:
            if abs(element[0] - element1[0]) < config["x_inbound"] and element != element1:
                legit.remove(element1)

    for i, element in enumerate(legit):
        if i != 0:
            prev = legit[i-1]
            if abs(element[0]-prev[0]) > config['x_offset']:
                legit.remove(element)

    for element in legit:
        (x, y, w, h) = element
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # new = image[legit[0][1] : legit[-1][1], legit[0][0]: legit[-1][0]]

    new = np.empty_like(image)
    # for element in legit:
    #     new[element[1]:element[1]+element[3], element[0]:element[0]+element[2]] = image[element[1]:element[1]+element[3], element[0]:element[0]+element[2]]

    if len(legit) > 0:
        new = image_gray[legit[0][1] - 30: legit[-1][1] + legit[-1][3] + 30, legit[0][0] - 30: legit[-1][0] + legit[-1][2] + 30]
        new = cv.equalizeHist(new)
        new = cv.bilateralFilter(new, 11, 17, 17)
        new = cv.adaptiveThreshold(new, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 25, 0)
        new = cv.morphologyEx(new, cv.MORPH_OPEN, kernel, iterations=1)
        new = cv.morphologyEx(new, cv.MORPH_GRADIENT, kernel, iterations=1)


    # new = cv.adaptiveThreshold(new, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 25, 25)





    cv.imshow("Original Image", image)
    cv.imshow('Processed', thresh)
    cv.imshow('new', new)
    # cv.imshow('crop', new)
    cv.waitKey(2000)




def perform_processing(image: np.ndarray) -> str:
    img = image
    img = cv.resize(img, (640, 480))
    extract_plate(img)
    return 'XXXXXXX'
