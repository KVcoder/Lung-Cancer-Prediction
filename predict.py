import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
from skimage.feature import hog
from skimage import io, transform, exposure
import os
import numpy as np
import pydicom
from PIL import Image

model = tf.keras.models.load_model('best_model_CNN.keras')
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)   
        if filename.lower().endswith('.dcm'):
            dicom = pydicom.dcmread('file_path')
            img_array = dicom.pixel_array
            if img_array.dtype != np.uint8:
                img_array = exposure.rescale_intensity(img_array, out_range='uint8')
        else:
            img = io.imread(file_path, as_gray=True)
            if img.dtype != np.uint8:
                img = (img * 255).astype('uint8')              
        img = Image.fromarray(img)
        img = img.convert('L') if img.mode != 'L' else img 
        img = np.array(img)    
        img = transform.resize(img, (100, 100), anti_aliasing=True)   
        images.append(img)
    return np.array(images)


root = tk.Tk()
root.withdraw()
file_path = filedialog.askdirectory()
images = load_images_from_folder(file_path) ### PASS FOLDER WITH DATA

X = images.reshape(-1, 100, 100, 1)

predictions = model.predict(X)
predicted_classes = np.array(tf.argmax(predictions, axis=1), dtype='U10')
predicted_classes[predicted_classes == '0'] = 'Benign'
predicted_classes[predicted_classes == '1'] = 'Malignant'
predicted_classes[predicted_classes == '2'] = 'Normal'
print(predicted_classes)


