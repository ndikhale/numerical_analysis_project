
from streamlit_drawable_canvas import st_canvas
from manual_editing.manual_edit import save_img_fft, save_fft_edits, get_mask_from_edits, masking
import numpy as np
import cv2
import streamlit as st
from PIL import Image
from fourier_transform.fourier import color_fft, color_ifft, fft2, ifft2, post_process_image, get_magnitude
from automated_filtering.automated_filter import gaussian_low_pass_filter, gaussian_high_pass_filter, idealFilter_low_pass_filter, \
    idealFilter_high_pass_filter, butterworth_low_pass_filter, butterworth_high_pass_filter

st.set_page_config(layout="wide")
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




def manual():
    st.title("Manual Filtering")

    if bg_image is not None:
        with st.form(key = "form"):
            isColor = False

            image = Image.open(bg_image)
            image = image.resize((256, 256))
            image_arr = np.array(image)

            if len(image_arr.shape) == 3:
                isColor = True

            col1, col2, col3 = st.columns(3)

            with col1:
                st.text("")

            with col2:
                st.image(image_arr, use_column_width=False)

            with col3:
                st.text("")

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
                    edited_img_r = create_canvas(fft_names[0], 'r', image_arr.shape[0], image_arr.shape[1])
                with col2:
                    st.text("Green Channel: ")
                    edited_img_g = create_canvas(fft_names[1], 'g', image_arr.shape[0], image_arr.shape[1])
                with col3:
                    st.text("Blue Channel: ")
                    edited_img_b = create_canvas(fft_names[2], 'b', image_arr.shape[0], image_arr.shape[1])
            else:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.text("")
                with col2:
                    st.text("FFT of Image: ")
                    edited_img_w = create_canvas(fft_names[0], 'w', image_arr.shape[0], image_arr.shape[1])
                with col3:
                    st.text("")

            if st.form_submit_button("Find Inverse Fourier Transform: "):
                if isColor:
                    edited_img = [edited_img_r.image_data, edited_img_g.image_data, edited_img_b.image_data]
                    edited_img_names = ["edited_img_r.png", "edited_img_g.png", "edited_img_b.png"]
                else:
                    edited_img = [edited_img_w.image_data]
                    edited_img_names = ["edited_img_w.png"]

                # save fft edits
                save_fft_edits(edited_img, edited_img_names)

                # get the edits
                fft_edits = []
                for i in range(len(edited_img_names)):
                    fft_edits.append(cv2.imread(edited_img_names[i], -1))

                # get mask
                masks = get_mask_from_edits(fft_edits)

                # apply masks
                masked_img = masking(image_fft, masks)

                # apply inverse tranform
                inverse_tranform_img = color_ifft(masked_img, isColor)
                cv2.imwrite("inverse_tranform_img.png", inverse_tranform_img)
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.text("")

                with col2:
                    st.image(inverse_tranform_img, use_column_width=False)

                with col3:
                    st.text("")




def create_canvas(background_image, key, height, width):

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 1)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=Image.open(background_image),
        update_streamlit=realtime_update,
        height=height,
        width=width,
        drawing_mode=drawing_mode,
        point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
        key=key,
    )

    return canvas_result


def automated():
    st.title("Automated Filtering")

    if bg_image is not None:
        with st.form(key="form"):
            isColor = False

            image = Image.open(bg_image)
            image = image.resize((256, 256))
            image_arr = np.array(image)

            print("image size: " + str(image_arr.shape))
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
                #if isColor:
                print("color: "+str(isColor))
                if filter_option == "Gaussian Low Pass Filter":
                    if isColor:
                        mask_result = []
                        for i in range(0, 3):
                            mask_result.append(gaussian_low_pass_filter(50, image_arr[:, :, i].shape))
                    else:
                        mask_result = gaussian_low_pass_filter(50, image_arr.shape)
                if filter_option == "Gaussian High Pass Filter":
                    if isColor:
                        mask_result = []
                        for i in range(3):
                            mask_result.append(gaussian_high_pass_filter(50, image_arr[:, :, i].shape))
                    else:
                        mask_result = gaussian_high_pass_filter(50, image_arr.shape)
                elif filter_option == "Ideal Low Pass Filter":
                    if isColor:
                        mask_result = []
                        for i in range(3):
                            mask_result.append(idealFilter_low_pass_filter(50, image_arr[:, :, i].shape))
                    else:
                        mask_result = idealFilter_low_pass_filter(50, image_arr.shape)
                elif filter_option == "Ideal High Pass Filter":
                    if isColor:
                        mask_result = []
                        for i in range(3):
                            mask_result.append(idealFilter_high_pass_filter(50, image_arr[:, :, i].shape))
                    else:
                        mask_result = idealFilter_high_pass_filter(50, image_arr.shape)
                elif filter_option == "Butter Worth Low Pass Filter":
                    if isColor:
                        mask_result = []
                        for i in range(3):
                            mask_result.append(butterworth_low_pass_filter(50, image_arr[:, :, i].shape, 1))
                    else:
                        mask_result = butterworth_low_pass_filter(50, image_arr.shape, 1)
                elif filter_option == "Butter Worth High Pass Filter":
                    if isColor:
                        mask_result = []
                        for i in range(3):
                            mask_result.append(butterworth_high_pass_filter(50, image_arr[:, :, i].shape, 1))
                    else:
                        mask_result = butterworth_high_pass_filter(50, image_arr.shape, 1)
                else:
                    if isColor:
                        mask_result = []
                        for i in range(3):
                            mask_result.append(gaussian_low_pass_filter(50, image_arr[:, :, i].shape))
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

                print("tranformed: " + str(fft_img))
                print("tranformed shape: " + str(fft_img.shape))
                cv2.imwrite("inverse_tranform_img.png", fft_img)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.text("")

                with col2:
                    st.image(fft_img.astype('int'), use_column_width=False)

                with col3:
                    st.text("")




def main():
    if "button_id" not in st.session_state:
        st.session_state["button_id"] = ""
    if "color_to_label" not in st.session_state:
        st.session_state["color_to_label"] = {}
    PAGES = {
        "Using Filter (Automated)": automated,
        "Edit the frequency (Manually)": manual,
    }
    page = st.sidebar.selectbox("Page:", options=list(PAGES.keys()))
    PAGES[page]()

    with st.sidebar:
        st.markdown("---")


if __name__ == "__main__":
    st.sidebar.subheader("Configuration")
    main()