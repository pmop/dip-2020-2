import numpy as np
import cv2
import math


# (1) Crop and flip an image using Numpy array indexing.
def crop_image(image, x, y, width, height):
    """
    :param image np array with the image in it's contents
    :type image: numpy.array
    :param x integer where the cropped image begins in the x axis, where the origin is the top left corner
    :param y integer where the cropped image begins in the y axis, where the origin is the top left corner
    :param width the cropped image final width
    :param height the cropped image final height
    :type height: int
    """
    return image[y:y+height, x:x+width]


def flip_image_horizontally(image):
    """
    :param image np array with the image in it's contents
    """
    return image[:, ::-1]


def flip_image_vertically(image):
    """
    :param image np array with the image in it's contents
    """
    return image[::-1, ...]


# (2) Implement image translation using Numpy and OpenCV.
def translate_image(image, tx, ty):
    image = to_brga(image)
    affine_matrix = np.float32([
        [1, 0, tx],
        [0, 1, ty]
    ])
    return cv2.warpAffine(image, affine_matrix, (image.shape[0], image.shape[1]))


def shear(cx, cy, angle):
    """
    :param cx: x-axis for the center of the rotation
    :param cy: y-axis for the center of the rotation
    :param angle: math.radians angle
    :return: array with shear x and y
    """
    tg = math.tan(angle/2)
    new_cx = round(cx - cy*tg)
    new_cy = cy
    new_cy = round(new_cx * math.sin(angle) + new_cy)
    new_cx = round(new_cx - new_cy*tg)
    return [new_cx, new_cy]


# (3) Implement image rotation using Numpy and OpenCV.
def rotate_image(image, angle):
    """
    :param image: source image
    :param angle: math.radians angle
    :return: np array with the rotated image
    """
    image = to_brga(image)

    cosine = math.cos(angle)
    sine   = math.sin(angle)
    height = image.shape[0]
    width  = image.shape[1]

    new_height = round(abs(image.shape[0] * cosine) + abs(image.shape[1] * sine)) + 1
    new_width  = round(abs(image.shape[1] * cosine) + abs(image.shape[0] * sine)) + 1

    output     = np.zeros((new_height, new_width, image.shape[2]))
    image_copy = output.copy()

    och = round(((image.shape[0] + 1) / 2) - 1)
    ocw = round(((image.shape[1] + 1) / 2) - 1)

    new_centre_height = round(((new_height + 1) / 2) - 1)
    new_centre_width = round(((new_width + 1) / 2) - 1)

    for i in range(height):
        for j in range(width):
            # co-ordinates of pixel with respect to the centre of original image
            y = image.shape[0] - 1 - i - och
            x = image.shape[1] - 1 - j - ocw
            # Applying shear Transformation
            new_y, new_x = shear(x, y, angle)
            new_y = new_centre_height - new_y
            new_x = new_centre_width - new_x

            output[new_y, new_x, :] = image[i, j, :]
    return output


# (4) Implement image resizing using OpenCV.
def resize_image(image, scale):
    """
    :param image the source image
    :type image: numpy.array
    :param scale the scale normalized
    :type scale: float
    """
    new_width = round(image.shape[0] * scale)
    new_height = round(image.shape[1] * scale)
    dim = (new_width, new_height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_LANCZOS4)


# (5) Implement bitwise operations: AND, OR, XOR.
def bitwise_and(image, mask):
    """
    :param image np array with the image in it's contents
    :param mask np array
    """
    rows = mask.shape[0]
    cols = mask.shape[1]
    result = np.copy(image)
    for x in range(0, cols - 1):
        for y in range(0, rows - 1):
            result[x, y] = result[x, y] & mask[x, y]
    return result


def bitwise_or(image, mask):
    """
    :param image np array with the image in it's contents
    :param mask np array
    """
    rows = mask.shape[0]
    cols = mask.shape[1]
    result = np.copy(image)
    for x in range(0, cols - 1):
        for y in range(0, rows - 1):
            result[x, y] = result[x, y] | mask[x, y]
    return result


def bitwise_xor(image, mask):
    """
    :param image np array with the image in it's contents
    :param mask np array
    """
    rows = mask.shape[0]
    cols = mask.shape[1]
    result = np.copy(image)
    for x in range(0, cols - 1):
        for y in range(0, rows - 1):
            result[x, y] = result[x, y] ^ mask[x, y]
    return result


def to_brga(image):
    height, width, channels = image.shape
    if channels < 4: image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    return image


# (6) Implement the "mask" operation, where a third image 'h' contains only a Region of Interest (ROI
# -- defined by the second image mask 'g') obtained from the input image 'f'. Note that this Region can be of any shape.
def mask_image_circle(image, cx, cy, cradius):
    """
    :param image np array with the image in it's contents
    :param cx integer circle x
    :param cy integer circle y
    :param cradius integer circle radius
    """
    image = to_brga(image)
    mask = np.zeros(image.shape, image.dtype)
    mask = cv2.circle(mask, [cx, cy], cradius, (255,)*image.shape[2], -1)
    return cv2.bitwise_and(image, mask)


def main():
    image = cv2.imread('./lena.png')
    iw = image.shape[0]
    ih = image.shape[1]
    """
    (1) Crop and flip an image using Numpy array indexing.
    (2) Implement image translation using Numpy and OpenCV.
    (3) Implement image rotation using Numpy and OpenCV.
    (4) Implement image resizing using OpenCV.
    (5) Implement bitwise operations: AND, OR, XOR.
    (6) Implement the "mask" operation, where a third image 'h' contains only a Region of Interest (ROI
        -- defined by the second image mask 'g') obtained from the input image 'f'.
         Note that this Region can be of any shape.
    """
    cropped_image = crop_image(image, 0, 0, round(iw/2), round(ih/2))
    cv2.imwrite('./lena-cropped.png', cropped_image)

    cropped_and_flipped_image_v = flip_image_vertically(cropped_image)
    cv2.imwrite('./lena-cropped-flipped-v.png', cropped_and_flipped_image_v)

    cropped_and_flipped_image_h = flip_image_horizontally(cropped_image)
    cv2.imwrite('./lena-cropped-flipped-h.png', cropped_and_flipped_image_h)

    translated_image = translate_image(image, 0, 200)
    cv2.imwrite('./lena-translated.png', translated_image)

    rotated_image = rotate_image(image, math.radians(-45))
    cv2.imwrite('./lena-rotated.png', rotated_image)

    resized_image = resize_image(image, 0.1)
    cv2.imwrite('./lena-resized.png', resized_image)

    # See in the code (5) Implement bitwise operations: AND, OR, XOR.
    roi_mask_image = mask_image_circle(image, round(iw/2), round(0.6 * ih), round(0.4 * iw))
    cv2.imwrite('./lena-roi_mask_image.png', roi_mask_image)


main()
