import os
from os import listdir

def read_files(folder):
# get the path/directory
      folder_dir = "C:/smartTech/ca1-smart-tech/tiny-imagenet-200/"+folder+"/images"
      images =[]
      for image in os.listdir(folder_dir):
            images.append(image)
      return images

if __name__ == "__main__":
      test_images = read_files("test")
      validation_images = read_files("val")
      print(len(validation_images))