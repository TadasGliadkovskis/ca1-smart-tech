import os
import cv2
from os import listdir
import env
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import torch
import torchvision
from torchvision.io import read_image



def read_ids():
	file_dir = env.IMAGES_DIR+"wnids.txt"
	with open(file_dir) as f:
		ids = f.readlines()
	return list(map(str.strip, ids))


def read_words_txt():
	file_dir = env.IMAGES_DIR+"words.txt"
	with open(file_dir) as f:
		lines = f.readlines()

	new_list = []
	ids = read_ids()
	for line in lines:
		split_id = line.split('\t')[0]
		if split_id in ids:
			new_list.append(line)
	return(new_list)


def filter_words_txt():
	file_dir = env.IMAGES_DIR+"filtered_words.txt"  # Output to this file
	dictionary_list = read_words_txt()
	with open(file_dir, 'w') as f:
		for line in dictionary_list:
			f.write(line)


def read_files(folder):
	images_dir = env.IMAGES_DIR+folder+"/images"
	images = []
	for file in os.listdir(images_dir):
		image = cv2.imread(os.path.join(images_dir, file))
		if image is not None:
			images.append(image)
	return images


def read_train_files(ids, limit):
	train = []
	for id in ids:
		train.append(read_files("/train/"+id))
		if limit:
			break
	return train


def grey_scale(image):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	return image


def gaus_blur(image):
	image = cv2.GaussianBlur(image, (5, 5), 0)
	return image


def equalise(image):
	image = cv2.equalizeHist(image)
	return image


def preprocess_image(image):
	image = grey_scale(image)
	image = gaus_blur(image)
	image = equalise(image)
	image = image/255
	return image

	
url = env.IMAGES_DIR+"train/n04540053/images/n04540053_0.JPEG"
def display_image(img, top_left_coords, bottom_right_coords):
	img = preprocess_image(img)
	img = img * 255
	img = img.astype(np.uint8)
	img_rect = cv2.rectangle(img, (int(top_left_coords[0]),int(top_left_coords[1])), (int(bottom_right_coords[0]),int(bottom_right_coords[1])), (255,0,0), 0)
	plt.imshow(img_rect, cmap='gray')
	plt.show()



# Will allow us to see how many classes (200), how many images in each class (500), (X, Y) resolution (64, 64), RGB status (3)
def get_shape(image_collection):
	print(np.shape(image_collection))
	return np.shape(image_collection)


def extract_bounding_box(id):
	txt_file = id+"_boxes.txt"
	boxes_dir = env.IMAGES_DIR+"train/"+id+"/"+txt_file
	with open(boxes_dir) as f:
		lines = f.readlines()
	# array of 500 lines going to 499
	split_list=[]
	top_left_coords = []
	bottom_right_coords = []
	for line in lines:
		split_list = line.split('\t')
		top_left_coords += [[split_list[1],split_list[2]]]
		bottom_right_coords +=[[split_list[3],split_list[4].strip()]]
	return top_left_coords, bottom_right_coords

	
if __name__ == "__main__":
	ids = read_ids()
	
	
	# Get bounding box co ordinates
	count = 0
	top_left = []
	bottom_right = []
	for id in ids:
		top_left, bottom_right = extract_bounding_box(id)

	print("top left: ",top_left)
	print("bottom right ",bottom_right)
	X_train = read_train_files(read_ids(),True)
	counter = 0
	for images in X_train:
		for image in images:
			display_image(image, top_left[counter], bottom_right[counter])			
			counter += 1
	
	# Get train data and display the first image from the first class
	# X_train = read_train_files(read_ids(),True)
	# X_test = read_files("test")
	# X_val = read_files("val")

	# X_train = np.array(list(map(preprocess_image, X_train)))
	# X_test = np.array(list(map(preprocess_image, X_test)))
	# X_val = np.array(list(map(preprocess_image, X_val)))
	# img = X_train[0][0]
	# print(len(X_train))
	
