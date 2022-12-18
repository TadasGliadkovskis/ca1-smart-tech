import os
import cv2
from os import listdir
import env
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from natsort import natsorted
import tensorflow.keras
import sklearn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.models import Model
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import pandas

def read_ids(id_file_name):
	file_dir = env.IMAGES_DIR+id_file_name
	with open(file_dir) as f:
		lines = f.readlines()
	ids = []
	for line in lines:
		id = line.split('\t')[0]
		ids.append(id.strip())
	return ids

def read_labels():
	file_dir = env.IMAGES_DIR+"filtered_words.txt"
	with open(file_dir) as f:
		lines = f.readlines()
	labels = []
	for line in lines:
		label = line.split('\t')[1]
		labels.append(label.strip())
	return labels

def read_words_txt():
	file_dir = env.IMAGES_DIR+"words.txt"
	with open(file_dir) as f:
		lines = f.readlines()

	new_list = []
	ids = read_ids("wnids.txt")
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
	files = os.listdir(images_dir)
	sorted_files = natsorted(files)
	for file in sorted_files:
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
		split_list = line.split('\t') 						# {"n01443537_0.JPEG", "0",	"10", "63", "58"}
		top_left_coords += [[split_list[1],split_list[2]]]
		bottom_right_coords +=[[split_list[3],split_list[4].strip()]]
		bounding_box = [top_left_coords, bottom_right_coords]
	return bounding_box

def modified_model():
	num_classes = 200
	model = Sequential()
	model.add(Conv2D(60, (5, 5), input_shape=(64, 64, 1), activation='relu'))
	model.add(Conv2D(60, (5, 5), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(30, (3, 3), activation='relu'))
	model.add(Conv2D(30, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(500, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
	return model


if __name__ == "__main__":
	
	y_train = read_labels()
	y_train = np.array(y_train)
	# print(y_train)
	
	# label_encoder = LabelEncoder()
	# y_train = label_encoder.fit_transform(y_train)
	# y_train = to_categorical(y_train)

	X_train = read_train_files(read_ids("filtered_words.txt"),False)
	X_test = read_files("test")
	X_val = read_files("val")

	X_train= X_train.reshape(X_train.shape[0], 64, 64, 1)


	# X_train = np.array(list(map(preprocess_image, X_train)))
	# X_test = np.array(list(map(preprocess_image, X_test)))
	# X_val = np.array(list(map(preprocess_image, X_val)))

	assert(get_shape(X_train)[0] == get_shape(y_train)[0])
	model = modified_model()
	#print(model.summary())

	y_train= pandas.get_dummies(y_train)

	y_train = to_categorical(y_train, 200)

	print(y_train)	
	# history = model.fit(X_train, y_train, validation_split=0.05, epochs=5, batch_size=200, verbose=1, shuffle=1)
	# plt.plot(history.history['loss'])
	# plt.plot(history.history['val_loss'])
	# plt.legend(['loss', 'val_loss'])
	# plt.title('Loss')
	# plt.xlabel('epochs')
	# plt.show()

	# Get bounding box co ordinates
	# count = 0
	# top_left = []
	# bounding_box = {}
	# #TODO this only retrieves the bounding box for the last class so when its used with display image
	# for id in ids:
	# 	bounding_box[id] = extract_bounding_box(id)
		
	# print(bounding_box['n02124075'][0][1], bounding_box['n02124075'][1][1])
	# print(bounding_box['n07749582'][0][1], bounding_box['n07749582'][1][1])

	
	# counter = 0
	# #TODO Can make this into a method and then use it in the loop above
	# for images in X_train:
	# 	for image in images:
	# 		display_image(image, top_left[counter], bottom_right[counter])			
	# 		counter += 1
	
	# Get train data and display the first image from the first class
	# X_train = read_train_files(read_ids(),True)



	# img = X_train[0][0]
	# print(len(X_train))
	
