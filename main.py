import os
import cv2
from os import listdir
import env


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

def make_new_words_txt():
	file_dir = env.IMAGES_DIR+"new_words.txt"
	dictionary_list = read_words_txt()
	with open(file_dir,'w') as f:
		for line in dictionary_list:
			f.write(line)

def read_files(folder):
	folder_dir = env.IMAGES_DIR+folder+"/images"
	images = []
	for file in os.listdir(folder_dir):
		image = cv2.imread(os.path.join(folder_dir, file))
		if image is not None:
			images.append(image)
	return images


def read_train_files(ids):
	train = []
	for id in ids:
		train.append(read_files("/train/"+id))
	return train


def grey_scale(image):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	return image


def equalise(image):
	image = cv2.equalizeHist(image)
	return image


def preprocess_image(image):
	image = grey_scale(image)
	image = equalise(image)
	image = image/255
	return image


if __name__ == "__main__":
	make_new_words_txt()
