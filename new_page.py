import base64
import json
import os
import re
import time
import uuid
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import math
import matplotlib.pyplot as plt
from svgpathtools import parse_path
st.set_page_config(layout="wide")
# Specify canvas parameters in application
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:",
    ("freedraw", "line", "rect", "circle", "transform", "polygon", "point"),
)
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
if drawing_mode == 'freedraw':
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg", "jpeg"])
realtime_update = st.sidebar.checkbox("Update in realtime", True)
#display_toolbar=st.sidebar.checkbox("Display toolbar", True)

def main():
    if "button_id" not in st.session_state:
        st.session_state["button_id"] = ""
    if "color_to_label" not in st.session_state:
        st.session_state["color_to_label"] = {}
    PAGES = {
        "Using Filter (Automated)": automated_filtering,
        "Edit the frequency (Manually)": manual,
    }
    page = st.sidebar.selectbox("Page:", options=list(PAGES.keys()))
    PAGES[page]()

    with st.sidebar:
        st.markdown("---")

def manual():
    st.title("Manual Filtering")
    #st.sidebar.header("Configuration")

    if bg_image is not None:
        with st.form(key = "form"):
            isColor = False

            image = Image.open(bg_image)
            image = image.resize((256, 256))
            image_arr = np.array(image)

            print("image size: "+str(image_arr.shape))
            if len(image_arr.shape) == 3:
                isColor = True

            col1, col2, col3 = st.columns(3)

            with col1:
                st.text("")

            with col2:
                st.image(image_arr, use_column_width=False)

            with col3:
                st.text("")

            #cv2.imwrite("sample_img.png", image_arr)
            image_fft = color_fft(image_arr, isColor)

            if isColor:
                fft_names = ["img_fft_r.png", "img_fft_g.png", "img_fft_b.png"]
            else:
                fft_names = ["img_fft.png"]

            save_img_fft(image_fft, fft_names)

            # show fft on screen
            if isColor:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.text("Red Channel: ")
                    canvas_r = create_canvas_draw_instance(fft_names[0], "red", image_arr.shape[0], image_arr.shape[1])
                with col2:
                    st.text("Green Channel: ")
                    canvas_g = create_canvas_draw_instance(fft_names[1], "green", image_arr.shape[0], image_arr.shape[1])
                with col3:
                    st.text("Blue Channel: ")
                    canvas_b = create_canvas_draw_instance(fft_names[2], "blue", image_arr.shape[0], image_arr.shape[1])
            else:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.text("")
                with col2:
                    st.text("FFT of Image: ")
                    canvas_img = create_canvas_draw_instance(fft_names[0], "gray", image_arr.shape[0], image_arr.shape[1])
                with col3:
                    st.text("")

            if st.form_submit_button("Find Inverse Fourier Transform: "):
                if isColor:
                    edited_img = [canvas_r.image_data, canvas_g.image_data, canvas_b.image_data]
                    edited_img_names = ["edited_img_r.png", "edited_img_g.png", "edited_img_b.png"]
                else:
                    edited_img = [canvas_img.image_data]
                    edited_img_names = ["edited_img.png"]

                # save fft edits
                save_fft_edits(edited_img, edited_img_names)

                # get the edits
                fft_edits = []
                for i in range(len(edited_img_names)):
                    fft_edits.append(cv2.imread(edited_img_names[i], -1))

                # get mask
                list_mask = get_mask_from_canvas(fft_edits)

                # apply masks
                masked_img = apply_mask_all(image_fft, list_mask)

                # apply inverse tranform
                inverse_tranform_img = color_ifft(masked_img, isColor)
                print("tranformed: "+str(inverse_tranform_img))
                print("tranformed shape: " + str(inverse_tranform_img.shape))
                cv2.imwrite("inverse_tranform_img.png", inverse_tranform_img)
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.text("")

                with col2:
                    st.image(inverse_tranform_img, use_column_width=False)

                with col3:
                    st.text("")

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
    return result

def automated_filtering():
    st.title("Automated Filtering")

    if bg_image is not None:
        with st.form(key = "form"):
            isColor = False

            image = Image.open(bg_image)
            image = image.resize((256, 256))
            image_arr = np.array(image)

            print("image size: "+str(image_arr.shape))
            if len(image_arr.shape) == 3:
                isColor = True

            col1, col2, col3 = st.columns(3)

            with col1:
                st.text("")

            with col2:
                st.image(image_arr, use_column_width=False)

            with col3:
                st.text("")

            filter_option = st.selectbox('Which filter would you like to use?',
            ('Gaussian Low Pass Filter',
            'Gaussian High Pass Filter',
            'Ideal Low Pass Filter',
            'Ideal High Pass Filter',
            'Butter Worth Low Pass Filter',
            'Butter Worth High Pass Filter'))

            if isColor:
                fft_names = ["img_fft_r.png", "img_fft_g.png", "img_fft_b.png"]
            else:
                fft_names = ["img_fft.png"]

            if st.form_submit_button("Perform Filter"):

                if filter_option == "Gaussian Low Pass Filter":
                    if isColor:
                        mask_result = []
                        for i in range(0, 3):
                            mask_result.append(gaussian_low_pass_filter(50,image_arr[:,:,i].shape))
                    else:
                        mask_result = gaussian_low_pass_filter(50, image_arr.shape)
                if filter_option == "Gaussian High Pass Filter":
                    if isColor:
                        mask_result = []
                        for i in range(3):
                            mask_result.append(gaussian_high_pass_filter(50,image_arr[:,:,i].shape))
                    else:
                        mask_result = gaussian_high_pass_filter(50, image_arr.shape)
                elif filter_option == "Ideal Low Pass Filter":
                    if isColor:
                        mask_result = []
                        for i in range(3):
                            mask_result.append(idealFilter_low_pass_filter(50,image_arr[:,:,i].shape))
                    else:
                        mask_result = idealFilter_low_pass_filter(50, image_arr.shape)
                elif filter_option == "Ideal High Pass Filter":
                    if isColor:
                        mask_result = []
                        for i in range(3):
                            mask_result.append(idealFilter_high_pass_filter(50,image_arr[:,:,i].shape))
                    else:
                        mask_result = idealFilter_high_pass_filter(50, image_arr.shape)
                elif filter_option == "Butter Worth Low Pass Filter":
                    if isColor:
                        mask_result = []
                        for i in range(3):
                            mask_result.append(butterworth_low_pass_filter(50,image_arr[:,:,i].shape, 1))
                    else:
                        mask_result = butterworth_low_pass_filter(50, image_arr.shape, 1)
                elif filter_option == "Butter Worth High Pass Filter":
                    if isColor:
                        mask_result = []
                        for i in range(3):
                            mask_result.append(butterworth_high_pass_filter(50,image_arr[:,:,i].shape, 1))
                    else:
                        mask_result = butterworth_high_pass_filter(50, image_arr.shape, 1)
                else:
                    if isColor:
                        mask_result = []
                        for i in range(3):
                            mask_result.append(gaussian_low_pass_filter(50,image_arr[:, :, i].shape))
                    else:
                        mask_result = gaussian_low_pass_filter(50, image_arr.shape)

                color_image_fft = 0
                image_fft = 0
                fft_img = 0

                if isColor:
                    color_image_fft = color_fft(image_arr, isColor)
                    images_fft = []
                    for i in range(3):
                        fft_img_new = color_image_fft[i] * mask_result[i]
                        images_fft.append(fft_img_new)
                        
                    fft_img = color_ifft(images_fft, isColor)
                else:
                    image_fft = fft2(image_arr)
                    multiple_result = image_fft * mask_result
                    inverse_tranform_img = ifft2(multiple_result)
                    fft_img = get_magnitude(inverse_tranform_img)
                    fft_img = post_process_image(fft_img)

                print("tranformed: "+str(fft_img))
                print("tranformed shape: " + str(fft_img.shape))
                cv2.imwrite("inverse_tranform_img.png", fft_img)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.text("")

                with col2:
                    st.image(fft_img.astype('int'), use_column_width=False)

                with col3:
                    st.text("")

def create_canvas_draw_instance(background_image, key, height, width):

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 1)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=Image.open(background_image),
        update_streamlit=realtime_update,
        height=height,
        width=width,
        drawing_mode=drawing_mode,
        point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
        #display_toolbar=st.sidebar.checkbox("Display toolbar", True, key=key),
        key=key,
    )

    return canvas_result

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
        temp1[r] = DFT_slow(a[r])
    print("done with first")
    for c in range(len(a[0])):
        temp2[:, c] = DFT_slow(temp1[:, c])
    return np.rint(temp2)

def DFT_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    #x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    #M = (1j*np.sin(-2 * math.pi * k * n / N) ) + np.cos(-2 * math.pi * k * n / N)
    M = np.cos(2 * math.pi * (k-(N/2)) * (n-(N/2)) / N) - (1j*np.sin(2 * math.pi * (k-(N/2)) * (n-(N/2)) / N))
    return np.dot(M, x)

def ifft2(a):
    a = np.asarray(a)
    temp1 = np.zeros(a.shape, complex)
    temp2 = np.zeros(a.shape, complex)
    for r in range(len(a)):
        temp1[r] = iDFT_slow(a[r])
    print("done with first")
    for c in range(len(a[0])):
        temp2[:, c] = iDFT_slow(temp1[:, c])
    return np.rint(temp2)

def iDFT_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    #x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    #M = (1j*np.sin(-2 * math.pi * k * n / N) ) + np.cos(-2 * math.pi * k * n / N)
    M = np.cos(2 * math.pi * (k-(N/2)) * (n-(N/2)) / N) + (1j*np.sin(2 * math.pi * (k-(N/2)) * (n-(N/2)) / N))
    return np.dot(M, x)

def save_img_fft(images, names):
    for image, name in zip(images, names):
        plt.imsave(name, np.log(1 + abs(image)), cmap='gray')

def save_fft_edits(image, name):
    for image, name in zip(image, name):
        cv2.imwrite(name, image)

def get_mask_from_canvas(canvas_images):
    list_mask = []
    for image in canvas_images:
        list_mask.append(image[:, :, 3])

    return list_mask

def apply_mask(input_image, mask):
    _, mask_thresh = cv2.threshold(mask, 120, 255, cv2.THRESH_BINARY)
    #cv2.imwrite("thresh.png", mask_thresh)
    mask_bool = mask_thresh.astype('bool')
    input_image[mask_bool] = 1
    # print("mask thresh shape: " + str(mask_thresh.shape))
    # print("mask thresh: " + str(mask_thresh))
    # for i in range(len(input_image)):
    #     for j in range(len(input_image[0])):
    #         input_image[i][j] = input_image[i][j] * mask_thresh[i][j]

    return input_image

def apply_mask_all(list_images, list_mask):
    final_result = []
    for (i, mask) in zip(list_images, list_mask):
        result = apply_mask(i, mask)
        final_result.append(result)
    return final_result

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

if __name__ == "__main__":
    # st.set_page_config(
    #     page_title="Streamlit Drawable Canvas Demo", page_icon=":pencil2:"
    # )
    # st.title("Drawable Canvas Demo")
    st.sidebar.subheader("Configuration")
    main()