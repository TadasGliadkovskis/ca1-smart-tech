import os
import cv2
from os import listdir
import env
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from natsort import natsorted
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.models import Model
from keras.utils.np_utils import to_categorical
import pandas
import random
import requests
from io import BytesIO
import zipfile

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
		label = line.split('\t')[0]
		for i in range(500):
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
		train += (read_files("/train/"+id))
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
	img = img * 255
	img = img.astype(np.uint8)
	img_rect = cv2.rectangle(img, (int(top_left_coords[0]),int(top_left_coords[1])), (int(bottom_right_coords[0]),int(bottom_right_coords[1])), (255,0,0), 0)
	plt.imshow(img_rect, cmap='gray')
	plt.show()


def display_random_images(X_train, amount):
	for i in range(amount):
		rand = random.randint(0, 100001)
		plt.imshow(X_train[rand], cmap = plt.get_cmap('gray'))			
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
	model.add(Conv2D(60, (5, 5), input_shape=(64, 64, 1), activation='elu'))
	model.add(Conv2D(60, (5, 5), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(30, (3, 3), activation='elu'))
	model.add(Conv2D(30, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(500, activation='elu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
	return model


def download_images():
	if (os.path.isdir(env.IMAGES_DIR)):
		print ('Images already downloaded...')
		return
	url="http://cs231n.stanford.edu/tiny-imagenet-200.zip"
	r = requests.get(url, stream=True)
	print ('Downloading |' + url )
	zip_ref = zipfile.ZipFile(BytesIO(r.content))
	zip_ref.extractall('./')
	zip_ref.close()

if __name__ == "__main__":
	
	download_images()
	filter_words_txt()
	print("Reading ID's")
	ids = read_ids("filtered_words.txt")
	print("Fetching train, test and val data This will take a while if the operation is run for the first time")
	X_train = read_train_files(ids,False)
	X_test = read_files("test")
	X_val = read_files("val")
	print("Before Preprocess shape")
	get_shape(X_train)
	X_train = np.array(list(map(preprocess_image, X_train)))

	X_train = X_train.reshape(100000, 64, 64, 1)

	print("After Preprocess shape")
	print(X_train.shape)

	y_train = read_labels()
	y_train= np.array(y_train)
	print(y_train)
	y_train = pandas.get_dummies(y_train)

	print("Y train shape")
	print(y_train.shape)
	# label_encoder = LabelEncoder()
	# y_train = label_encoder.fit_transform(y_train)
	# y_train = to_categorical(y_train)

	
	

	TOP_LEFT_COORDS = 0
	BOTTOM_RIGHT_COORDS = 1
	bounding_box = {}
	
	for id in ids:
		bounding_box[id] = extract_bounding_box(id)

	#shows an image the boundary box
	display_image(X_train[500], bounding_box['n01629819'][TOP_LEFT_COORDS][0], bounding_box['n01629819'][BOTTOM_RIGHT_COORDS][0])			
	display_random_images(X_train, 3)
	
	# X_test = np.array(list(map(preprocess_image, X_test)))
	# X_val = np.array(list(map(preprocess_image, X_val)))

	model = modified_model()
	#print(model.summary())

	
	# Train the model and evaluate its performance
	h = model.fit(X_train, y_train, batch_size=200, epochs=50, validation_split=0.2,  verbose=1, shuffle=1)
	plt.plot(h.history['accuracy'])
	plt.plot(h.history['val_accuracy'])
	plt.title('Accuracy')
	plt.xlabel('epoch')
	plt.show()

	model.save(env.IMAGES_DIR+"../model.h5")

	# DOES NOT WORK 
	# url=env.IMAGES_DIR+"/tiny-imagenet-200/test/images/test_0.JPEG"
	# img = cv2.imread(url)
	# img = np.asarray(img)
	# img = cv2.resize(img, (32, 32))
	# img = preprocess_image(img)
	# img = img.reshape(1, 32, 32, 1)
	# print("Predicted sign: " + str(np.argmax(model.predict(img), axis=1)))
