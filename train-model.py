import tensorflow as tf
import os #navigate through directories
import cv2
import imghdr
from matplotlib import pyplot as plt
import numpy as np

#Avoid OOM errors by setting GPU memory growth
gpus = tf.config.experimental.list_physical_devices('CPU')
        
data_dir = 'data'

image_exts = ['jpg', 'jpeg', 'png', 'bmp']


# for image_class in os.listdir(data_dir):
#     for image in os.listdir(os.path.join(data_dir, image_class)):
#         print(image)

# img = cv2.imread(os.path.join('data', 'happy', 'Friends-at-an-Event.jpg'))

# print(img.shape)
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()


for image_class in os.listdir(data_dir):
    if image_class == '.DS_Store':
        continue
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print(f'Invalid image: {image_path}')
                os.remove(image_path)
        except Exception as e :
            print('Issue with image {}'.format(image_path))
            
data = tf.keras.utils.image_dataset_from_directory('data')

data_iterator = data.as_numpy_iterator() 

batch = data_iterator.next()


