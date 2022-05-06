import numpy as np
import math


def calculate_distance(location_1, location_2):
    return math.sqrt((location_1[0] - location_2[0]) ** 2 + (location_1[1] - location_2[1]) **2)


def gaussian_low_pass_filter(dist, image_shape):
    result = np.zeros(image_shape[:2])
    rows, cols = image_shape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            result[y, x] = math.exp(((-calculate_distance((y, x), center) ** 2) / (2 * (dist ** 2))))
    return result


def gaussian_high_pass_filter(dist, image_shape):
    result = np.zeros(image_shape[:2])
    rows, cols = image_shape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            result[y, x] = 1 - math.exp(((-calculate_distance((y, x), center) ** 2) / (2 * (dist ** 2))))
    return result


def idealFilter_low_pass_filter(dist, image_shape):
    result = np.zeros(image_shape[:2])
    rows, cols = image_shape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            if calculate_distance((y, x), center) < dist:
                result[y, x] = 1
    return result


def idealFilter_high_pass_filter(dist, image_shape):
    result = np.ones(image_shape[:2])
    rows, cols = image_shape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            if calculate_distance((y, x), center) < dist:
                result[y, x] = 0
    return result


def butterworth_low_pass_filter(dist, image_shape, n):
    result = np.zeros(image_shape[:2])
    rows, cols = image_shape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            result[y, x] = 1 / (1 + (calculate_distance((y, x), center) / dist) ** (2 * n))
    return result


def butterworth_high_pass_filter(dist, image_shape, n):
    result = np.zeros(image_shape[:2])
    rows, cols = image_shape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            result[y, x] = 1 - 1 / (1 + (calculate_distance((y, x), center) / dist) ** (2 * n))
    return