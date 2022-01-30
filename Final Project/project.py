from pickle import NONE
import cv2 as cv
import numpy as np
import sys
import matplotlib.pyplot as plt
import pytesseract
from matplotlib.pyplot import contour
capture = cv.VideoCapture('modified.mp4')

count = 1


# def find_contours(dimensions, img):

#     # Find all contours in the image
#     cntrs, _ = cv.findContours(
#         img.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

#     # Retrieve potential dimensions
#     lower_width = dimensions[0]
#     upper_width = dimensions[1]
#     lower_height = dimensions[2]
#     upper_height = dimensions[3]

#     # Check largest 5 or  15 contours for license plate or character respectively
#     cntrs = sorted(cntrs, key=cv.contourArea, reverse=True)[:15]

#     ii = cv.imread('contour.jpg')

#     x_cntr_list = []
#     target_contours = []
#     img_res = []
#     for cntr in cntrs:
#         # detects contour in binary image and returns the coordinates of rectangle enclosing it
#         intX, intY, intWidth, intHeight = cv.boundingRect(cntr)

#         # checking the dimensions of the contour to filter out the characters by contour's size
#         if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height:
#             # stores the x coordinate of the character's contour, to used later for indexing the contours
#             x_cntr_list.append(intX)

#             char_copy = np.zeros((44, 24))
#             # extracting each character using the enclosing rectangle's coordinates.
#             char = img[intY:intY+intHeight, intX:intX+intWidth]
#             char = cv.resize(char, (20, 40))

#             cv.rectangle(ii, (intX, intY), (intWidth+intX,
#                          intY+intHeight), (50, 21, 200), 1)
#             #plt.imshow(ii, cmap='gray')
#             rev = pytesseract.image_to_string(ii)
#             print(rev)
#             # Make result formatted for classification: invert colors
#             char = cv.subtract(255, char)

#             # Resize the image to 24x44 with black border
#             char_copy[2:42, 2:22] = char
#             char_copy[0:2, :] = 0
#             char_copy[:, 0:2] = 0
#             char_copy[42:44, :] = 0
#             char_copy[:, 22:24] = 0

#             # List that stores the character's binary image (unsorted)
#             img_res.append(char_copy)

#     # Return characters on ascending order with respect to the x-coordinate (most-left character first)

#     # plt.show()
#     # function that stores sorted list of character indeces
#     indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
#     img_res_copy = []
#     for idx in indices:
#         # stores character images according to their index
#         img_res_copy.append(img_res[idx])
#     img_res = np.array(img_res_copy)

#     return img_res


def segment_characters(image):

    # Preprocess cropped license plate image
    img_lp = cv.resize(image, (333, 75))
    img_gray_lp = cv.cvtColor(img_lp, cv.COLOR_BGR2GRAY)
    _, img_binary_lp = cv.threshold(
        img_gray_lp, 200, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    img_binary_lp = cv.erode(img_binary_lp, (3, 3))
    img_binary_lp = cv.dilate(img_binary_lp, (3, 3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    # Make borders white
    img_binary_lp[0:3, :] = 255
    img_binary_lp[:, 0:3] = 255
    img_binary_lp[72:75, :] = 255
    img_binary_lp[:, 330:333] = 255

    # Estimations of character contours sizes of cropped license plates
    dimensions = [LP_WIDTH/6,
                  LP_WIDTH/2,
                  LP_HEIGHT/10,
                  2*LP_HEIGHT/3]
    #plt.imshow(img_binary_lp, cmap='gray')
    # plt.show()
    #cv.imwrite('contour.jpg', img_binary_lp)
    rev = pytesseract.image_to_string(img_binary_lp)
    fileobj = open('output.txt', 'w')
    if(len(rev) > 10):
        for i in range(len(rev)):
            if(rev[i].isalnum() or rev[i] == ' '):
                print(rev[i], end='')
                fileobj.write(rev[i])
        fileobj.close()
        sys.exit(0)
    # Get contours within cropped license plate
    #char_list = find_contours(dimensions, img_binary_lp)

   # return char_list


def preprocess(img):  # find_possible_plate()   ->  #main

    # img = cv.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
    blur = cv.GaussianBlur(img, (7, 7), 0)
    # cv.imshow('blurr', blur)

    gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
    # cv.imshow('gray', gray)

    sobelx = cv.Sobel(gray, cv.CV_8U, 1, 0, ksize=3)

    ret2, otsu_thresh = cv.threshold(
        sobelx, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # cv.imshow('otsu', otsu_thresh)

    element = cv.getStructuringElement(cv.MORPH_RECT, (22, 5))
    morph_n_thresh = cv.morphologyEx(otsu_thresh, cv.MORPH_CLOSE, element)
    # cv.imshow('morph_n_thresh', morph_n_thresh)
    return morph_n_thresh


def extract_contours(after_preprocess):  # find_possible_plate()   ->  #main

    contours, _ = cv.findContours(
        after_preprocess, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours


def ratioCheck(area, width, height):

    min = 4500
    max = 30000

    ratioMin = 3
    ratioMax = 6

    ratio = float(width) / float(height)

    if ratio < 1:
        ratio = 1 / ratio

    if (area < min or area > max) or (ratio < ratioMin or ratio > ratioMax):
        return False

    return True


def clean_plate(plate):
    gray = cv.cvtColor(plate, cv.COLOR_BGR2GRAY)
    thresh = cv.adaptiveThreshold(
        gray,  255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    contours, _ = cv.findContours(
        thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if contours:
        areas = []
        for c in contours:
            areas.append(cv.contourArea(c))

        max_index = np.argmax(areas)

        max_cnt = contours[max_index]
        max_cntArea = areas[max_index]
        x, y, w, h = cv.boundingRect(max_cnt)
        rect = cv.minAreaRect(max_cnt)

        if not ratioCheck(max_cntArea, plate.shape[1],
                          plate.shape[0]):
            return plate, False, None
        return plate, True, [x, y, w, h]

    else:
        return plate, False, None


def check_plate(img, cnts):  # find_possible_plate()   ->  #main
    min_rect = cv.minAreaRect(cnts)

    if validateRatio(min_rect):
        x, y, w, h = cv.boundingRect(cnts)
        after_validation_img = img[y:y+h, x:x+w]  # cropping the image
        if(after_validation_img.shape[1] > after_validation_img.shape[0]):
            #cv.imshow('after_validation_img', after_validation_img)
            after_clean_plate_img, plateFound, coordinates = clean_plate(
                after_validation_img)
            if plateFound:

                for c in cnts:
                    # approximate the contour
                    peri = cv.arcLength(c, True)
                    approx = cv.approxPolyDP(c, 0.20 * peri, True)
                    # if our approximated contour has four points, then
                    # we can assume that we have found our screen
                    screenCnt = 0
                    if len(approx) == 4:
                        screenCnt = approx
                        break

                    if screenCnt is not NONE:
                        char = segment_characters(after_clean_plate_img)

    return None, None, None


def find_possible_plate(img):  # main function
    plates = []
    char_on_plate = []
    corresponding_area = []

    # returns a morphed and thersholded image
    after_preprocess = preprocess(img)

    possible_plate_contours = extract_contours(
        after_preprocess)  # returns contours which is a list

    for cnts in possible_plate_contours:
        plate, characters_on_plate, coordinates = check_plate(img, cnts)

        if plate is not None:
            plates.append(plate)
            char_on_plate.append(characters_on_plate)
            corresponding_area.append(coordinates)


# validateRatio() ->  #checkplate() ->  #find_possible_plate()   ->  #main
def preRatioCheck(area, width, height):
    min = 4500
    max = 30000

    ratioMin = 2.5
    ratioMax = 7

    ratio = float(width)/float(height)

    if ratio < 1:
        ratio = 1/ratio

    if (area < min or area > max) or (ratio < ratioMin or ratio > ratioMax):
        return False

    return True


def validateRatio(min_rect):  # checkplate() ->  #find_possible_plate()   ->  #main
    (x, y), (w, h), rect_angle = min_rect

    if(w > h):
        angle = -rect_angle
    else:
        angle = rect_angle+90

    if(angle > 15):
        return False

    if(h == 0 or w == 0):
        return False

    area = w*h

    if not preRatioCheck(area, w, h):
        return False
    else:
        return True


def main():
    count = 1
    start_frame_number = 50

    while True:
        ret, frame = capture.read()
        if ret:
            #cv.imshow('original video', frame)
            find_possible_plate(frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        capture.set(cv.CAP_PROP_POS_FRAMES, start_frame_number)
        start_frame_number += 11


if __name__ == "__main__":
    main()
