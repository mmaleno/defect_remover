"""
Filename:       defect_remover.py
Creator:        Max Maleno
Date:           Due 2020/05/05
Description:    Final Project for MS160 at Scripps College.
                Removes defects from scanned images.
Usage: python3 defect_remover.py threshold_px <path_to_first_image> <path_to_second_image>
"""

import cv2
from matplotlib import pyplot as plt
import numpy as np
import sys
import time

def load_images(filepath_im1, filepath_im2):
    im1 = cv2.imread(filepath_im1)
    im2 = cv2.imread(filepath_im2)
    return im1, im2

def show_images(images, win_name = None, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
       Adapted from Prof. Douglas Goodwin's show_images function.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    win_name: name of window to be displayed

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """

    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]

    if win_name is None: fig = plt.figure()
    else: fig = plt.figure(win_name)

    for n, (image, title) in enumerate(zip(images, titles)):

        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
        else:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        a.set_title(title)
        plt.axis('off')
    plt.show()

def align_images(im_secondary, im_base):
    """Align the secondary image with the base image.
       Outputs the aligned secondary image and the matches diagram.
       Adapted from Learn OpenCV's tutorial:
       https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
    
    Parameters
    ---------
    im_secondary: image to be aligned
    
    im_base: mostly square image
    """

    MAX_FEATURES = 100000
    GOOD_MATCH_PERCENT = 0.20

    # Convert images to grayscale
    im_secondary_gray = cv2.cvtColor(im_secondary, cv2.COLOR_BGR2GRAY)
    im_base_gray = cv2.cvtColor(im_base, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    # ORB is the best to use because it's freeee!
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im_secondary_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im_base_gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im_secondary, keypoints1, im_base, keypoints2, matches, None)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im_base.shape
    if h is not None:
        im1Reg = cv2.warpPerspective(im_secondary, h, (width, height))
    else:
        im1Reg = im_secondary

    return im1Reg, imMatches

def detect_differences(img1, img2, th):
    """Detect the differences between images img1 and img2 at threshold th.
       Outputs the common image with (0,0,0)s where differences are, the 
       image that represents the locations of the differences, and the list
       of detected difference contours.
        
    Parameters
    ---------
    img1: first OpenCV image object to be compared

    img2: second OpenCV image object to be compared (already aligned)
    
    th: integer threshold value to detect differences
    """
    # create different templates for the differences
    diff_orig = cv2.absdiff(img1, img2)
    mask = cv2.cvtColor(diff_orig, cv2.COLOR_BGR2GRAY)
    imask =  mask>int(th)
    canvas_im1 = np.zeros_like(img1, np.uint8)
    canvas_im1[imask] = img1[imask]

    # merge template and original image to maybe come back to
    diff_1 = cv2.absdiff(img1, canvas_im1)
    mask = cv2.cvtColor(diff_1, cv2.COLOR_BGR2GRAY)
    th = 1
    imask =  mask>th
    canvas_holes = np.zeros_like(img1, np.uint8)
    canvas_holes[imask] = img2[imask]

    # diff image to create contours from
    diff_image = cv2.cvtColor(canvas_im1, cv2.COLOR_BGR2GRAY)

    # generate contours for differences in diff_image
    contours = cv2.findContours(diff_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    return canvas_holes, diff_image, contours

def remove_bad_contours(img1, img2, contours, canvas_holes):
    """Remove detected differences, determine which image has the real data
    for each difference, and combine the detected real data with the base image.
    Outputs the calculated image with real data in place of the detected differences.

    Parameters
    ---------
    img1: first OpenCV image object to be compared

    img2: second OpenCV image object to be compared (already aligned)
    
    contours: list of contour coordinates from detect_differences()

    canvas_holes: OpenCV image object of base image with (0,0,0)s at difference coordinates
    """
    mask = np.zeros(img1.shape, np.uint8)
    canvas_filled = np.zeros(img1.shape, dtype='uint8')

    for c in contours:
        area = cv2.contourArea(c)
        if area > 40:
            mask[...] = 0
            x,y,w,h = cv2.boundingRect(c)
            cv2.drawContours(mask, [c], 0, (255,255,255), -1)
            roi_1_avg = cv2.mean(img1[y:y+h,x:x+w])[0]
            contour_1_avg = cv2.mean(img1[c],mask[c])[0]
            roi_2_avg = cv2.mean(img2[y:y+h,x:x+w])[0]
            contour_2_avg = cv2.mean(img2[c],mask[c])[0]

            if np.abs(roi_1_avg - contour_1_avg) < np.abs(roi_2_avg - contour_2_avg):
                #print("Img1 should keep this contour")
                canvas_filled = cv2.bitwise_or(canvas_filled, cv2.bitwise_and(img1, mask))
            else:
                #print("Img1 should take this contour from Img2")
                canvas_filled = cv2.bitwise_or(canvas_filled, cv2.bitwise_and(img2, mask))

            cv2.rectangle(img1, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.rectangle(img2, (x, y), (x + w, y + h), (36,255,12), 2)
            
    combined = cv2.bitwise_or(canvas_holes, canvas_filled)

    return combined, canvas_filled, img1, img2


def main():
    print("\nStarting defect remover\n")

    im_base, im_secondary = load_images(sys.argv[2], sys.argv[3])
    print("Images loaded\n")
    
    im_secondary_rotated, im_matches = align_images(im_secondary, im_base)

    im_base_copy = im_base.copy()
    im_secondary_rotated_copy = im_secondary_rotated.copy() # to save before ROIs are added

    base_holes, diff_image, contours = detect_differences(im_base, im_secondary_rotated, sys.argv[1])

    output, contour_img, img1_boxes, img2_boxes = remove_bad_contours(im_base, im_secondary_rotated, contours, base_holes)

    print("Displaying images...\n")
    show_images([im_base_copy,im_secondary,im_matches,im_secondary_rotated_copy,base_holes,diff_image,img1_boxes,img2_boxes, \
    output,contour_img], \
                win_name = 'All Steps', cols = 5, titles = ['Base Image', 'Secondary Image', \
                'Matches Diagram', 'Rotated Secondary', 'Base Template with Holes', 'Difference Image', \
                'Base Image w/ ROIs', 'Secondary Image w/ ROIs', 'Output Image', 'All Contours'])

    print("Writing output image...")
    datetime = str(time.localtime().tm_year) + "_" + \
            str(time.localtime().tm_mon).zfill(2) + "_" + \
            str(time.localtime().tm_mday).zfill(2) + "_" + \
            str(time.localtime().tm_hour).zfill(2) + "_" + \
            str(time.localtime().tm_min).zfill(2) + "_" + \
            str(time.localtime().tm_sec).zfill(2)
    
    cv2.imwrite("output_"+datetime+sys.argv[2][sys.argv[2].find('.'):],cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    print("\nDone!\n")

if __name__ == '__main__':
    main()
