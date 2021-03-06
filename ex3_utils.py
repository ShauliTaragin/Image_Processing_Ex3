import math
import sys
from typing import List

import numpy as np
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 209337161


# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------

def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10,
                win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """
    # If the image is not grey scale then change it to grey scale
    if len(im1.shape) > 2:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    if len(im2.shape) > 2:
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    # find the derivative to x and derivative from y
    vector = np.array([[1, 0, -1]])
    I_X = cv2.filter2D(im2, -1, vector, borderType=cv2.BORDER_REPLICATE)
    I_Y = cv2.filter2D(im2, -1, vector.T, borderType=cv2.BORDER_REPLICATE)
    I_T = im2 - im1

    # initialize returning arrays which is the u_v we find and the x_y is the point we check on
    u_v = []
    x_y = []
    # maybe look to start loop from winsize/2 and not step size
    for i in range(step_size, im1.shape[0], step_size):
        for j in range(step_size, im1.shape[1], step_size):
            # create the small sample out of ix,iy,it and work on it
            sample_I_X = I_X[i - win_size // 2:i + win_size // 2 + 1, j - win_size // 2: j + win_size // 2 + 1]
            sample_I_Y = I_Y[i - win_size // 2:i + win_size // 2 + 1, j - win_size // 2: j + win_size // 2 + 1]
            sample_I_T = I_T[i - win_size // 2:i + win_size // 2 + 1, j - win_size // 2: j + win_size // 2 + 1]

            # flatten the sample we are working on since we enter it in the matrix as a vector
            sample_I_X = sample_I_X.flatten()
            sample_I_Y = sample_I_Y.flatten()
            sample_I_T = sample_I_T.flatten()

            # size of samples
            n = len(sample_I_X)

            # I will now calculate the (A^tA)^-1 and A^tB matrices
            sum_IX_squared = sum(sample_I_X[h] ** 2 for h in range(n))
            sum_IX_IY = sum(sample_I_X[h] * sample_I_Y[h] for h in range(n))
            sum_IY_squared = sum(sample_I_Y[h] ** 2 for h in range(n))

            sum_IX_IT = sum(sample_I_X[h] * sample_I_T[h] for h in range(n))
            sum_IY_IT = sum(sample_I_Y[h] * sample_I_T[h] for h in range(n))

            # Enter what we calculated into a 2x2 matrix
            A = np.array([[sum_IX_squared, sum_IX_IY], [sum_IX_IY, sum_IY_squared]])
            B = np.array([[-sum_IX_IT], [-sum_IY_IT]])  # check this shape

            # get eigen values
            eigen_val, eigen_vec = np.linalg.eig(A)
            eig_val1 = eigen_val[0]
            eig_val2 = eigen_val[1]

            # find that the eigen values hold the conditions we set out for them
            # find largest one
            if eig_val1 < eig_val2:
                temp = eig_val1
                eig_val1 = eig_val2
                eig_val2 = temp

            # condition 2 if it holds add the point
            if eig_val2 <= 1 or eig_val1 / eig_val2 >= 100:
                continue

            # calculate u and v
            vector_u_v = (np.linalg.inv(A)) @ B
            u = vector_u_v[0][0]
            v = vector_u_v[1][0]

            x_y.append([j, i])
            u_v.append([u, v])

    return np.array(x_y), np.array(u_v)


def opticalFlowPyrLK(img1: np.ndarray, img2: np.ndarray, k: int,
                     stepSize: int, winSize: int):
    """
    :param img1: First image
    :param img2: Second image
    :param k: Pyramid depth
    :param stepSize: The image sample size
    :param winSize: The optical flow window size (odd number)
    :return: A 3d array, with a shape of (m, n, 2),
    where the first channel holds U, and the second V.
    """

    uv_return = []
    xy_return = []
    img1_pyr = gaussianPyr(img1, k)
    img2_pyr = gaussianPyr(img2, k)
    # entering the last pyramid
    x_y_prev, u_v_prev = opticalFlow(img1_pyr[-1], img2_pyr[-1], stepSize, winSize)
    x_y_prev = list(x_y_prev)
    u_v_prev = list(u_v_prev)
    for i in range(1, k):
        # find optical flow for this level
        x_y_i, uv_i = opticalFlow(img1_pyr[-1 - i], img2_pyr[-1 - i], stepSize, winSize)
        uv_i = list(uv_i)
        x_y_i = list(x_y_i)
        for g in range(len(x_y_i)):
            x_y_i[g] = list(x_y_i[g])
            # uv_i[g] = list(uv_i[g])
        # update uv according to formula
        for j in range(len(x_y_prev)):
            x_y_prev[j] = [element * 2 for element in x_y_prev[j]]
            u_v_prev[j] = [element * 2 for element in u_v_prev[j]]
        # If location of movements we found are new then append them, else add them to the proper location
        for j in range(len(x_y_i)):
            if x_y_i[j] in x_y_prev:
                u_v_prev[j] += uv_i[j]
            else:
                x_y_prev.append(x_y_i[j])
                u_v_prev.append(uv_i[j])
    # now we shall change uv and xy to a 3 dimensional array
    arr3d = np.zeros(shape=(img1.shape[0], img1.shape[1], 2))
    for x in range(img1.shape[0]):
        for y in range(img1.shape[1]):
            if [y, x] not in x_y_prev:
                arr3d[x, y] = [0, 0]
            else:
                arr3d[x, y] = u_v_prev[x_y_prev.index([y, x])]
    return arr3d


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """
    xy, uv = opticalFlow(im1, im2, step_size=20, win_size=5)
    # basically we will iterate over all the u,v's we got and check which one gives the best result i.e the MSE
    u = uv[:, [0]]
    u = list(u.T[0])
    u = np.array(u)
    # u_average = np.median(u)
    v = uv[:, [1]]
    v = list(v.T[0])
    v = np.array(v)
    min_difference = sys.maxsize
    translation_mat = np.array([[0, 0, 0],
                                [0, 0, 0],
                                [0, 0, 0]], dtype=np.float)
    for i in range(len(u)):
        t_ui = u[i]
        t_vi = v[i]
        # create the mat with current uv
        translation_mat_i = np.array([[1, 0, t_ui],
                                      [0, 1, t_vi],
                                      [0, 0, 1]], dtype=np.float)
        # create the image with current uv
        img_i = cv2.warpPerspective(im1, translation_mat_i, im1.shape[::-1])
        # find the mse
        mse = np.square(im2 - img_i).mean()
        # check whether current mse is least and update accordingly
        if mse < min_difference:
            min_difference = mse
            translation_mat = translation_mat_i

    return translation_mat


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """
    xy, uv = opticalFlow(im1, im2, 20, 5)
    xy_after_change = xy.copy()
    angle_list = []
    xy_after_change = xy_after_change.astype(float)
    for i in range(len(xy)):
        xy_after_change[i] += uv[i]
        angle_list.append(find_ang(xy[i], (0, 0), xy_after_change[i]))
    angle_list = np.array(angle_list)
    theta = np.median(angle_list)
    mat_to_extract_xy_from = findTranslationCorr(im1, im2)
    t_x = mat_to_extract_xy_from[0][2]
    t_y = mat_to_extract_xy_from[1][2]
    translation_mat = np.float32([[np.cos(np.radians(theta)), -np.sin(np.radians(theta)), t_x],
                                  [np.sin(np.radians(theta)), np.cos(np.radians(theta)), t_y],
                                  [0, 0, 1]])

    return translation_mat


def findXsYsCorr(pic1, pic2):
    """
    :param pic1: input image 1 in grayscale format.
    :param pic2: image 1 after Translation.
    :return: X's and Y's to find correlation.
    """
    subtle_pading = np.max(pic1.shape) // 2
    pading1 = np.fft.fft2(np.pad(pic1, subtle_pading))
    pading2 = np.fft.fft2(np.pad(pic2, subtle_pading))
    prod = pading1 * pading2.conj()
    result_full = np.fft.fftshift(np.fft.ifft2(prod))
    corr = result_full.real[1 + subtle_pading:-subtle_pading + 1, 1 + subtle_pading:-subtle_pading + 1]
    y1, x1 = np.unravel_index(np.argmax(corr), corr.shape)
    y2, x2 = np.array(pic2.shape) // 2
    return x1, y1, x2, y2


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """

    x_1, y_1, x_2, y_2 = findXsYsCorr(im1, im2)
    return np.float32([[1, 0, x_2 - x_1 - 1], [0, 1, y_2 - y_1 - 1], [0, 0, 1]])


def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """
    x_1, y_1, x_2, y_2 = findXsYsCorr(im1, im2)
    theta = find_ang((x_2, y_2), (0, 0), (x_1, y_1))
    mat = np.float32([
        [np.cos(np.radians(theta)), -np.sin(np.radians(theta)), 0],
        [np.sin(np.radians(theta)), np.cos(np.radians(theta)), 0],
        [0, 0, 1]
    ])
    mat = np.linalg.inv(mat)
    rotate = cv2.warpPerspective(im2, mat, im2.shape[::-1])
    x, y, x2, y2 = findXsYsCorr(im1, rotate)
    t_x = x2 - x - 1
    t_y = y2 - y - 1
    return np.float32([
        [np.cos(np.radians(theta)), -np.sin(np.radians(theta)), t_x],
        [np.sin(np.radians(theta)), np.cos(np.radians(theta)), t_y / 6],
        [0, 0, 1]
    ])


def find_ang(first, second: (0, 0), third, /):
    first_angles_y, first_angles_x = first[0] - second[0], first[1] - second[1]
    seconed_angles_y, seconed_angles_x = third[0] - second[0], third[1] - second[1]
    arctan1 = math.atan2(first_angles_x, first_angles_y)
    arctan2 = math.atan2(seconed_angles_x, seconed_angles_y)
    if arctan1 < 0: arctan1 += math.pi
    if arctan2 < 0: arctan2 += math.pi
    if arctan1 <= arctan2:
        return arctan2 - arctan1
    else:
        return math.pi / 3 + arctan2 - arctan1


def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """
    # initialize returning image
    ret_img = np.zeros_like(im2)

    # iterate over image 2
    for x in range(im2.shape[0]):
        for y in range(im2.shape[1]):
            # change the 2d pixel to 3d homagraphicaly
            pixel_3d = np.array([[x],
                                 [y],
                                 [1]])
            get_pixel_from_img1 = T @ pixel_3d
            img1_x = get_pixel_from_img1[0] / get_pixel_from_img1[2]
            img1_y = get_pixel_from_img1[1] / get_pixel_from_img1[2]

            # check if pixels are ints or floats
            float_x = img1_x % 1
            float_y = img1_y % 1

            # if they are float transform them from im2 according to formula
            if float_x != 0 or float_y != 0:
                ret_img[x, y] = ((1 - float_x) * (1 - float_y) * im2[int(np.floor(img1_x)), int(np.floor(img1_y))]) \
                                + (float_x * (1 - float_y) * im2[int(np.ceil(img1_x)), int(np.floor(img1_y))]) \
                                + (float_x * float_y * im2[int(np.ceil(img1_x)), int(np.ceil(img1_y))]) \
                                + ((1 - float_x) * float_y * im2[int(np.floor(img1_x)), int(np.ceil(img1_y))])
            # if they are ints transform them as is
            else:
                img1_x = int(img1_x)
                img1_y = int(img1_y)
                ret_img[x, y] = im2[img1_x, img1_y]
    return ret_img


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------
#
#
def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    img = img[0: np.power(2, levels) * int(img.shape[0] / np.power(2, levels)),
          0: np.power(2, levels) * int(img.shape[1] / np.power(2, levels))]
    pyr = [img]
    ker_size = 5
    for i in range(1, levels):
        kernel = cv2.getGaussianKernel(ker_size, 0.3 * ((ker_size - 1) * 0.5 - 1) + 0.8)
        img = cv2.filter2D(img, -1, kernel=kernel, borderType=cv2.BORDER_REPLICATE)
        img = cv2.filter2D(img, -1, kernel=np.transpose(kernel), borderType=cv2.BORDER_REPLICATE)
        img = img[::2, ::2]
        pyr.append(img)
    return pyr


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    pyr = []
    ker_size = 5
    # similar to gaussian kernel however we will round the sigma to int inorder to receive a more
    # vivid picture
    kernel = cv2.getGaussianKernel(ker_size, int(0.3 * ((ker_size - 1) * 0.5 - 1) + 0.8))
    kernel = (kernel * kernel.transpose()) * 4
    gaussian_pyr = gaussianPyr(img, levels)
    for i in range(levels - 1):
        pyr_img = gaussian_pyr[i + 1]
        extended_pic = np.zeros((pyr_img.shape[0] * 2, pyr_img.shape[1] * 2))
        extended_pic[::2, ::2] = pyr_img
        extend_level = cv2.filter2D(extended_pic, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        curr_level = gaussian_pyr[i] - extend_level
        pyr.append(curr_level)
    pyr.append(gaussian_pyr[-1])
    return pyr


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Restores the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    ker_size = 5
    lap_pyr_copy = lap_pyr.copy()
    kernel = cv2.getGaussianKernel(ker_size, int(0.3 * ((ker_size - 1) * 0.5 - 1) + 0.8))
    kernel = (kernel * kernel.transpose()) * 4
    cur_layer = lap_pyr[-1]
    for i in range(len(lap_pyr_copy) - 2, -1, -1):
        extended_pic = np.zeros((cur_layer.shape[0] * 2, cur_layer.shape[1] * 2))
        extended_pic[::2, ::2] = cur_layer
        cur_layer = cv2.filter2D(extended_pic, -1, kernel, borderType=cv2.BORDER_REPLICATE) + lap_pyr_copy[i]
    return cur_layer


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray,
             mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """

    img_1 = img_1[0: np.power(2, levels) * int(img_1.shape[0] / np.power(2, levels)),
            0: np.power(2, levels) * int(img_1.shape[1] / np.power(2, levels))]
    img_2 = img_2[0: np.power(2, levels) * int(img_2.shape[0] / np.power(2, levels)),
            0: np.power(2, levels) * int(img_2.shape[1] / np.power(2, levels))]
    mask = mask[0: np.power(2, levels) * int(mask.shape[0] / np.power(2, levels)),
           0: np.power(2, levels) * int(mask.shape[1] / np.power(2, levels))]

    im_blend = np.zeros(img_1.shape)
    # check if the image is RGB
    if len(img_1.shape) > 2 or len(img_2.shape) > 2:
        for intensity in range(3):
            part_im1 = img_1[:, :, intensity]
            part_im2 = img_2[:, :, intensity]
            part_mask = mask[:, :, intensity]
            lp_reduce1 = laplaceianReduce(part_im1, levels)
            lp_reduce2 = laplaceianReduce(part_im2, levels)
            gauss_pyr = gaussianPyr(part_mask, levels)
            lp_ret = []
            for i in range(levels):
                curr_lup = gauss_pyr[i] * lp_reduce1[i] + (1 - gauss_pyr[i]) * lp_reduce2[i]
                lp_ret.append(curr_lup)
            im_blend[:, :, intensity] = laplaceianExpand(lp_ret)

    else:
        lp_reduce1 = laplaceianReduce(img_1, levels)
        lp_reduce2 = laplaceianReduce(img_2, levels)
        gauss_pyr = gaussianPyr(mask, levels)
        lp_ret = []
        for i in range(levels):
            curr_lup = gauss_pyr[i] * lp_reduce1[i] + (1 - gauss_pyr[i]) * lp_reduce2[i]
            lp_ret.append(curr_lup)
        im_blend = laplaceianExpand(lp_ret)

    # According to formula from TA's presentation do Naive blend
    naive_blend = mask * img_1 + (1 - mask) * img_2

    return naive_blend, im_blend
