import numpy as np
import cv2
import matplotlib.pyplot as plt

def save_img_fft(ffts, names):
    for fft, name in zip(ffts, names):
        plt.imsave(name, np.log(1 + abs(fft)), cmap='gray')

def save_fft_edits(images, names):
    for image, name in zip(images, names):
        cv2.imwrite(name, image)

def get_mask_from_edits(images):
    masks = []
    for image in images:
        masks.append(image[:, :, 3])
    return masks



def mask_fft(fft, mask):
    _, thresholded_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    thesh_bool = thresholded_mask.astype('bool')
    fft[thesh_bool] = 1
    return fft


def masking(ffts, masks):
    final_result = []
    for (fft, mask) in zip(ffts, masks):
        result = mask_fft(fft, mask)
        final_result.append(result)
    return final_result
