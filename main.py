import os
import cv2
from os import listdir
import env

def read_files(folder):
# get the path/directory
      folder_dir = env.IMAGES_DIR+folder+"/images"
      images =[]
      for file in os.listdir(folder_dir):
            image = cv2.imread(os.path.join(folder_dir,file))
            if image is not None:
                  images.append(image)
      return images

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
      test_images = read_files("test")
      validation_images = read_files("val")
      print(len(validation_images))