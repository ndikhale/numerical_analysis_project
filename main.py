import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template, send_file
from werkzeug.utils import secure_filename
import numpy as np
import math
import cv2
from matplotlib import image

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

@app.route('/success')
def success():
	return render_template('success.html', filename=request.args.get('filename'), my_name = request.args.get('my_name'), filter_image_name = request.args.get('filter_image_name'))

def rotate_filter(filter):
	rows = filter.shape[0]
	cols = filter.shape[1]
	rotated_filter = np.zeros((rows, rows))

	for i in range(0, rows):
		for j in range(0, cols):
			rotated_filter[i, rows-1-j] = filter[rows-1-i, j]

	return rotated_filter

def get_gaussian_filter():
    """Initialzes/Computes and returns a 5X5 Gaussian filter"""
    gaussianFilter = np.zeros((5, 5), dtype=float)
    sigma = 1.4

    for i in range(-2,3):
        for j in range(-2,3):
            gaussianFilter[i+2][j+2] = ((1) / (2 * np.pi * math.pow(sigma, 2))) * (np.exp(- (((i**2) + (j**2)) / (2 * math.pow(sigma, 2)))))

    sum = 0.0
    for i in range(0,5):
        for j in range(0,5):
            sum = sum + gaussianFilter[i][j]

    for i in range(0,5):
        for j in range(0,5):
            gaussianFilter[i][j] = (gaussianFilter[i][j]) / sum

    return gaussianFilter
	
def calculate_gaussian(npimg):
	output_array = np.zeros(npimg.shape)
	guassianFilter = get_gaussian_filter()

	rotatedFilter = rotate_filter(guassianFilter)

	image_rows, image_cols = npimg.shape[0], npimg.shape[1]
	filter_rows, filter_cols = rotatedFilter.shape

	padding_rows = int((filter_rows - 1) / 2)
	padding_cols = int((filter_cols - 1) / 2)

	padded_image = np.zeros((image_rows + (2 * padding_rows), image_cols + (2 * padding_cols)))

	padded_image[padding_rows:padded_image.shape[0] - padding_rows, padding_cols:padded_image.shape[1] - padding_cols] = npimg

	for row in range(image_rows):
		for col in range(image_cols):
			sum = 0
			for i in range(0, filter_rows):
				for j in range(0, filter_cols):
					sum = sum + (rotatedFilter[i][j] * padded_image[row+i][col+j])
			output_array[row,col] = sum

	return output_array

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():

	my_name = request.form.get("filters")
	print(my_name)

	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']

	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		flash('Image successfully uploaded and displayed below')

		npimg = image.imread("static/uploads/" + file.filename)

		result_array = calculate_gaussian(npimg)

		output_dir = 'static/uploads/'
		filter_image_name = "result_gaussian.jpg"
		output_image_name = output_dir  + filter_image_name
		cv2.imwrite(output_image_name, result_array)

		#return send_file('static/uploads/result_gaussian.jpg', as_attachment=True)
		return redirect(url_for('success',filename=filename, my_name = my_name, filter_image_name = filter_image_name))
		#return render_template('upload.html', filename=filename, my_name = my_name, filter_image_name = filter_image_name)
	else:
		flash('Allowed image types are -> png, jpg, jpeg')
		return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/download')
def download():
   return send_file('static/uploads/result_gaussian.jpg', as_attachment=True)

if __name__ == "__main__":
    app.run()