from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np

IMG_WIDTH, IMG_HEIGHT = 224, 224

def load_and_preprocess_image(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT)):
    img = keras_image.load_img(img_path, target_size=target_size)
    img_array = keras_image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded), img_array

def load_segmentation_mask(mask_path, target_size=(IMG_WIDTH, IMG_HEIGHT)):
    mask = keras_image.load_img(mask_path, target_size=target_size, color_mode="grayscale")
    mask_array = keras_image.img_to_array(mask) / 255.0
    mask_array_binary = (mask_array > 0.5).astype(np.uint8)
    return mask_array_binary