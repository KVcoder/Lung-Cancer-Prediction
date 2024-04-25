
import tensorflow as tf
from skimage.feature import hog
from skimage import io, transform
from sklearn.model_selection import train_test_split
import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
%matplotlib inline

def load_images_from_folder(folder):
    images = []
    labels = []
    for label, label_folder in enumerate(['Benign cases', 'Malignant cases', 'Normal cases']):
        path = os.path.join(folder, label_folder)
        for filename in os.listdir(path):
            img_path = os.path.join(path, filename)
            img = io.imread(img_path, as_gray=True)
            if img is not None:
                img = transform.resize(img, (100, 100))             
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

data_folder = 'The IQ-OTHNCCD lung cancer dataset'
images, labels = load_images_from_folder(data_folder)
X = images.reshape(-1, 100, 100, 1)
y = to_categorical(labels, num_classes=3)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
checkpoint = ModelCheckpoint(
    'best_model_CNN.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), callbacks=[checkpoint])
model.evaluate(X_test, y_test)




