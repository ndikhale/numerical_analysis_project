import numpy as np
import math

def color_fft(image, isColor):
    images_fft = []
    if isColor:
        for i in range(3):
            fft_img = fft2(image[:, :, i])
            images_fft.append(fft_img)
    else:
        images_fft.append(fft2(image))
    return images_fft

def color_ifft(image, isColor):
    images_ifft = []
    for i in range(len(image)):
        fft_img = ifft2(image[i])
        fft_img = get_magnitude(fft_img)
        fft_img = post_process_image(fft_img)
        images_ifft.append(fft_img)
    if isColor:
        resulted_img = np.dstack((images_ifft[0].astype('int'), images_ifft[1].astype('int'), images_ifft[2].astype('int')))
    else:
        resulted_img = np.array(images_ifft[0]).astype('int')
    return resulted_img

def fft2(a):
    a = np.asarray(a)
    temp1 = np.zeros(a.shape, complex)
    temp2 = np.zeros(a.shape, complex)
    for r in range(len(a)):
        temp1[r] = run_dft(a[r])
    for c in range(len(a[0])):
        temp2[:, c] = run_dft(temp1[:, c])
    return np.rint(temp2)

def run_dft(x):
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.cos(2 * math.pi * (k-(N/2)) * (n-(N/2)) / N) - (1j*np.sin(2 * math.pi * (k-(N/2)) * (n-(N/2)) / N))
    return np.dot(M, x)

def ifft2(a):
    a = np.asarray(a)
    temp1 = np.zeros(a.shape, complex)
    temp2 = np.zeros(a.shape, complex)
    for r in range(len(a)):
        temp1[r] = run_idft(a[r])
    for c in range(len(a[0])):
        temp2[:, c] = run_idft(temp1[:, c])
    return np.rint(temp2)

def run_idft(x):
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.cos(2 * math.pi * (k-(N/2)) * (n-(N/2)) / N) + (1j*np.sin(2 * math.pi * (k-(N/2)) * (n-(N/2)) / N))
    return np.dot(M, x)


def post_process_image(image):
    # perform full contrast stretch
    A = np.min(image)
    B = np.max(image)
    k = 256
    P = (k - 1) / (B - A)
    L = (-1 * A) * P
    contrased_image = [[0 for i in range(len(image[0]))] for j in range(len(image))]
    for i in range(len(image)):
        for j in range(len(image[0])):
            contrased_image[i][j] = (P * image[i][j]) + L

    return np.array(contrased_image)

def get_magnitude( image):
    final_mat = [[0 for i in range(len(image[0]))] for j in range(len(image))]
    real_mat = image.real
    imag_mat = image.imag

    for i in range(len(final_mat)):
        for j in range(len(final_mat[0])):
            final_mat[i][j] = (real_mat[i][j] ** 2) + (imag_mat[i][j] ** 2)
            final_mat[i][j] = round(math.sqrt(final_mat[i][j]))

    return np.array(final_mat)