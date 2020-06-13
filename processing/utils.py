import numpy as np
import cv2 as cv
import imutils
from scipy import ndimage
from skimage.measure import compare_ssim
from processing import sort_contours
from skimage.measure import compare_ssim as ssim


def reference():
    ref_A_J = cv.imread('C:/Users/witse/PycharmProjects/project_template_2020/Reference_A_J.png')
    ref_K_U = cv.imread('C:/Users/witse/PycharmProjects/project_template_2020/Reference_K_U.png')
    ref_V_Z = cv.imread('C:/Users/witse/PycharmProjects/project_template_2020/Reference_V_Z.png')
    ref_0_9 = cv.imread('C:/Users/witse/PycharmProjects/project_template_2020/Reference_0_9.png')
    digits = {}
    letters = {}
    all = {}
    letters_A_J = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    letters_K_U = ['K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U']
    letters_V_Z = ['V', 'W', 'X', 'Y', 'Z']

    ref_A_J = cv.cvtColor(ref_A_J, cv.COLOR_BGR2GRAY)
    ref_A_J = cv.threshold(ref_A_J, 10, 255, cv.THRESH_BINARY_INV)[1]
    refCnts_A_J = cv.findContours(ref_A_J.copy(), cv.RETR_EXTERNAL,
                                  cv.CHAIN_APPROX_SIMPLE)
    refCnts_A_J = imutils.grab_contours(refCnts_A_J)
    refCnts_A_J = sort_contours.sort_contours(refCnts_A_J, method="left-to-right")[0]

    ref_K_U = cv.cvtColor(ref_K_U, cv.COLOR_BGR2GRAY)
    ref_K_U = cv.threshold(ref_K_U, 10, 255, cv.THRESH_BINARY_INV)[1]
    refCnts_K_U = cv.findContours(ref_K_U.copy(), cv.RETR_EXTERNAL,
                                  cv.CHAIN_APPROX_SIMPLE)
    refCnts_K_U = imutils.grab_contours(refCnts_K_U)
    refCnts_K_U = sort_contours.sort_contours(refCnts_K_U, method="left-to-right")[0]

    ref_V_Z = cv.cvtColor(ref_V_Z, cv.COLOR_BGR2GRAY)
    ref_V_Z = cv.threshold(ref_V_Z, 10, 255, cv.THRESH_BINARY_INV)[1]

    refCnts_V_Z = cv.findContours(ref_V_Z.copy(), cv.RETR_EXTERNAL,
                                  cv.CHAIN_APPROX_SIMPLE)
    refCnts_V_Z = imutils.grab_contours(refCnts_V_Z)
    refCnts_V_Z = sort_contours.sort_contours(refCnts_V_Z, method="left-to-right")[0]

    ref_0_9 = cv.cvtColor(ref_0_9, cv.COLOR_BGR2GRAY)
    ref_0_9 = cv.threshold(ref_0_9, 10, 255, cv.THRESH_BINARY_INV)[1]
    refCnts_0_9 = cv.findContours(ref_0_9.copy(), cv.RETR_EXTERNAL,
                                  cv.CHAIN_APPROX_SIMPLE)
    refCnts_0_9 = imutils.grab_contours(refCnts_0_9)
    refCnts_0_9 = sort_contours.sort_contours(refCnts_0_9, method="left-to-right")[0]

    for (i, c) in enumerate(refCnts_0_9):
        (x, y, w, h) = cv.boundingRect(c)
        roi = ref_0_9[y:y + h, x:x + w]
        roi = cv.resize(roi, (65, 45))
        digits[i] = roi

    for (i, c) in enumerate(refCnts_A_J):
        (x, y, w, h) = cv.boundingRect(c)
        roi = ref_A_J[y:y + h, x:x + w]
        roi = cv.resize(roi, (65, 45))
        letters[letters_A_J[i]] = roi

    for (i, c) in enumerate(refCnts_K_U):
        (x, y, w, h) = cv.boundingRect(c)
        roi = ref_K_U[y:y + h, x:x + w]
        roi = cv.resize(roi, (65, 45))
        letters[letters_K_U[i]] = roi

    for (i, c) in enumerate(refCnts_K_U):
        (x, y, w, h) = cv.boundingRect(c)
        roi = ref_K_U[y:y + h, x:x + w]
        roi = cv.resize(roi, (65, 45))
        letters[letters_K_U[i]] = roi
    for (i, c) in enumerate(refCnts_V_Z):
        (x, y, w, h) = cv.boundingRect(c)
        roi = ref_V_Z[y:y + h, x:x + w]
        roi = cv.resize(roi, (65, 45))
        letters[letters_V_Z[i]] = roi
    all.update(letters)
    all.update(digits)
    return all


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

def process_plate(image):
    locs = []
    significant = []
    correctly_placed_cnts = {}
    x_prox = 30
    y_prox = 30

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
                correctly_placed_cnts[(x, y, w, h)] = contour

    significant.sort(key=lambda x: x[0])

    for i, element in enumerate(significant):
        if i != 0 and i != len(significant) - 1:
            (x, y, w, h) = element
            (x_p, y_p, w_p, h_p) = significant[i - 1]
            (x_n, y_n, w_n, h_n) = significant[i + 1]
            if x - (x_p + w_p) > x_prox and x_n - (x + w) > x_prox:
                significant.remove(element)
                correctly_placed_cnts.pop(element)

        if i == 0:
            (x, y, w, h) = element
            (x_n, y_n, w_n, h_n) = significant[i + 1]
            if x_n - (x + w) > x_prox:
                significant.remove(element)
                correctly_placed_cnts.pop(element)

        if i == len(significant) - 1:
            (x, y, w, h) = element
            (x_p, y_p, w_p, h_p) = significant[i - 1]
            if x - (x_p + w_p) > x_prox:
                significant.remove(element)
                correctly_placed_cnts.pop(element)

    for i, element in enumerate(significant):
        if i != 0 and i != len(significant) - 1:
            (x, y, w, h) = element
            (x_p, y_p, w_p, h_p) = significant[i - 1]
            (x_n, y_n, w_n, h_n) = significant[i + 1]
            if abs(y - y_p) > y_prox and abs(y_n - y) > y_prox:
                significant.remove(element)
                correctly_placed_cnts.pop(element)

        if i == 0:
            (x, y, w, h) = element
            (x_n, y_n, w_n, h_n) = significant[i + 1]
            if abs(y_n - y) > y_prox:
                significant.remove(element)
                correctly_placed_cnts.pop(element)

        if i == len(significant) - 1:
            (x, y, w, h) = element
            (x_p, y_p, w_p, h_p) = significant[i - 1]
            if abs(y - y_p) > y_prox:
                significant.remove(element)
                correctly_placed_cnts.pop(element)

    # for element in significant:
    #     (x, y, w, h) = element
    #     cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # cv.imshow('Plate', image)
    # cv.imshow('Plate_Canny', image_thresh)
    # cv.waitKey(1000)
    return significant, correctly_placed_cnts

def deskew(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.bitwise_not(gray)
    thresh = cv.threshold(gray, 0, 255,
                           cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(image, M, (w, h),
                             flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
    return rotated

def process_letters(letter_list, letters_cnts, plate, reference_dict):
    results = {}
    plate_read = []
    for i, element in enumerate(letter_list):
        kernel = np.ones((3, 3), dtype=np.uint8)
        letter_cnt = letters_cnts[element]
        letter = cv.minAreaRect(letter_cnt)
        box = cv.boxPoints(letter)
        box = np.int0(box)
        width = int(letter[1][0])
        height = int(letter[1][1])
        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height ],
                            [0, 0],
                            [width , 0],
                            [width , height]], dtype="float32")

        M = cv.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv.warpPerspective(plate, M, (width, height))
        if warped.shape[1] > warped.shape[0]:
            warped = cv.rotate(warped, cv.ROTATE_90_COUNTERCLOCKWISE)
        warped = deskew(warped)
        warped = cv.resize(warped, (65, 45))
        warped_gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
        warped_gray = cv.GaussianBlur(warped_gray, (5,5), 0)
        warped_thresh = cv.threshold(warped_gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
        warped_thresh = cv.dilate(warped_thresh, kernel)
        warped_thresh = cv.erode(warped_thresh, kernel)
        best_match = 0
        best_letter = None
        for (char, ROI) in reference_dict.items():
            s = ssim(warped_thresh, ROI, multichannel=False)
            if s > best_match:
                best_match = s
                best_letter = str(char)
        if (i == 0 or i == 1) and best_letter == '0':
            best_letter = 'O'
        plate_read.append(str(best_letter))
    return ''.join(plate_read)

        # cv.imshow('Letter', warped_thresh)
        # cv.imshow('Best Letter', reference_dict[best_letter])
        # cv.waitKey(2000)
        # cv.imshow('ROI', ROI)
        # cv.waitKey(2000)
        # cv.destroyAllWindows()


def perform_processing(image: np.ndarray) -> str:
    img = image
    img = cv.resize(img, (640, 480))
    reference_dict = reference()
    img_thresh = extract_plate(img)
    plate = save_plate(img_thresh, img)
    letter_list, letter_cnts = process_plate(plate)
    plate_read = process_letters(letter_list, letter_cnts, plate, reference_dict)

    return str(plate_read)
