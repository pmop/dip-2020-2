import numpy as np
import imutils
import cv2
import logging


def crop_border(image, size=10):
    cmb_image = cv2.copyMakeBorder(image, size, size, size, size,
                                   cv2.BORDER_CONSTANT, (0, 0, 0))
    gray = cv2.cvtColor(cmb_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    mask = np.zeros(thresh.shape, dtype="uint8")
    (x, y, w, h) = cv2.boundingRect(c)
    mask = cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    # TODO: fix bug
    min_rect = mask.copy()
    sub = mask.copy()
    while cv2.countNonZero(sub) > 0:
        min_rect = cv2.erode(min_rect, None)
        sub = cv2.subtract(min_rect, thresh)
    cnts = cv2.findContours(min_rect.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(c)
    return image[y:y + h, x:x + w]



def image_stitch(list_of_images):
    stitcher = cv2.Stitcher_create()
    return stitcher.stitch(list_of_images)


def main():
    logging.basicConfig(level=logging.INFO)
    imgs = []
    logging.info('Reading images...')
    for img_path in ['./ic1.jpg', './ic2.jpg', './ic3.jpg', './ic4.jpg']:
        imgs = [*imgs, cv2.imread(img_path)]
    logging.info('Beginning stitching...')
    (status, stitched) =  image_stitch(imgs)
    # logging.info('Stitching finished. Cropping the border...')
    # final_image = crop_border(stitched)
    # cv2.imwrite('./ic-stitched.jpg', final_image)
    logging.info('Stitching finished.')
    cv2.imwrite('./ic-stitched.jpg', stitched)


main()