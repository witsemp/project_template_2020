import numpy as np
import cv2 as cv
import imutils
class Detector():
    def __init__(self, config=None, image=None, debug=False):
        if not config:
            # Default na iets wat wat min of meer sal werk
            config = {"y_offset": 20,  # maximum y offset between chars
                      "x_offset": 55,  # maximum x gap between chars
                      "thesh_offset": 0,  # this determines the cutoff point on the adaptive threshold.
                      "thesh_window": 25,  # window of adaptive theshold area
                      # max min char width, height and ratio
                      "w_min": 6,  # char pixel width min
                      "w_max": 30,  # char pixel width max
                      "h_min": 12,  # char pixel height min
                      "h_max": 40,  # char pixel height max
                      "hw_min": 1.5,  # height to width ration min
                      "hw_max": 3.5,  # height to width ration max
                      "h_ave_diff": 1.09,  # acceptable limit for variation between characters
                      }
        self.setConfig(config)
        self.plates = []
        self.image = image
        self.thesh = None
        self.debug = debug

    def setConfig(self, config):
        self.config = config

    def detect_plates(self, image=None, level=1):
        if image is not None:
            self.image = image
        # First convert to black ad white
        image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)

        # many magic values in here, the thresh offset is around 0, as thresholding is done for a sliding window of 25x25 pixels
        self.thresh = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV,
                                            self.config["thesh_window"], self.config["thesh_offset"])
        if self.debug:
            self.allblobs = self.image.copy()
            self.reducedblobs = self.image.copy()
            self.roiblobs = self.image.copy()
        self.image = image
        # self.thesh = thresh
        # now we have a blck and white image and need to find all blobs that match our size and aspect requirements
        # find contours (This acts like a CCA) scikit seems to have a CCA as well, I just happended to find this first.
        # merge requests are welcome... with proof of higher accurace or quicker execution
        (cnts, _) = cv.findContours(self.thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # loop over the contours and discard all odd shapes and sizes
        correctly_sized_list = []
        for c in cnts:
            (x, y, w, h) = cv.boundingRect(c)
            if self.debug:
                cv.rectangle(self.allblobs, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Filter on width and height
            if self.config["w_min"] < w < self.config["w_max"] and self.config["h_min"] < h < self.config["h_max"] and \
                    self.config["hw_min"] < 1.0 * h / w < self.config["hw_max"]:
                correctly_sized_list.append((x, y, w, h))
                if self.debug:
                    cv.rectangle(self.reducedblobs, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # now we try to filter based on character proximity and the fact that they would be in a row
        self.possible_plate_regions = []
        # sort by x position
        self.sort_list = sorted(correctly_sized_list, key=lambda x: x[0])

        # Try to group blobs into platelike groups
        for char in self.sort_list:
            placed_char = False
            # Check if this blob has same y and x within offset values off current are (this is why we sorted by x value).
            for region in self.possible_plate_regions:
                if region[-1][1] - self.config["y_offset"] < char[1] < region[-1][1] + self.config["y_offset"] and \
                        region[-1][0] + self.config["x_offset"] > char[0]:
                    region.append(char)
                    placed_char = True
                    break
            # if char was not placed in a group, it becomes the first of a new group
            if placed_char is False:
                self.possible_plate_regions.append([char])

        # Now remove chars from regions if heights differ significantly, as numberplate chars are evenly sized. This could possibly be done in above filter, but this seemed better
        self.possible_plate_regions_ave_filtered = []

        for region in self.possible_plate_regions:
            if len(region) > 2:
                self.possible_plate_regions_ave_filtered.append([])
                ave = sum([char[3] for char in region]) / len(region)
                for char in region:
                    if ave / self.config["h_ave_diff"] < char[3] < ave * self.config["h_ave_diff"]:
                        self.possible_plate_regions_ave_filtered[-1].append(char)

        # Now filter char regions on count
        self.possible_plate_regions_ave_filtered = [x for x in self.possible_plate_regions_ave_filtered if len(x) > 2]

        possible_plate_regions_plate_details = []

        for region in self.possible_plate_regions_ave_filtered:
            # Find the min and max values of the plate region
            xmin = min([x[0] for x in region])
            ymin = min([x[1] for x in region])
            xmax = max([x[0] + x[2] for x in region])
            ymax = max([x[1] + x[3] for x in region])
            topleft = sorted(region, key=lambda x: x[0] + x[1])[0]
            topright = sorted(region, key=lambda x: -(x[0] + x[2]) + x[1])[0]
            botleft = sorted(region, key=lambda x: x[0] - (x[1] + x[3]))[0]
            botright = sorted(region, key=lambda x: -(x[0] + x[2]) - (x[1] + x[3]))[0]

            # print (topleft, topright, botleft, botright)

            mtop = 1.0 * (topleft[1] - topright[1]) / (topleft[0] - (topright[0] + topright[2]))
            mbot = 1.0 * (botleft[1] + botleft[3] - (botright[1] + botright[3])) / (
                        botleft[0] - (botright[0] + botright[2]))
            # print mtop, mbot
            if self.debug:
                for char in region:
                    (x, y, w, h) = char
                    cv.rectangle(self.roiblobs, (x, y), (x + w, y + h), (0, 0, 255), 1)

            possible_plate_regions_plate_details.append({"size": (xmin, ymin, xmax, ymax),
                                                         "roi": (
                                                         xmin - 2 * self.config["w_max"], ymin - self.config["h_max"],
                                                         xmax + 2 * self.config["w_max"], ymax + self.config["h_max"]),
                                                         "average_angle": (mtop + mbot) / 2.0})
            # Get area plus 2 x max char width to the sides and max char height above and below
            try:
                self.skew_correct(possible_plate_regions_plate_details[-1])

                # use thresholded roi to find chars again
                if "warped2" in possible_plate_regions_plate_details[-1] and possible_plate_regions_plate_details[-1][
                    "warped2"] is not None:
                    self.detect_chars(possible_plate_regions_plate_details[-1])
                    if len(possible_plate_regions_plate_details[-1]["plate"]) > 3:
                        possible_plate_regions_plate_details[-1]["somechars"] = True
            except Exception as ex:
                print
                ex

        self.plates = possible_plate_regions_plate_details
        return self.plates