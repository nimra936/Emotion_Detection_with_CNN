# import required packages
import cv2
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# ---------------- GPU CONFIG ----------------
# Check if TensorFlow can see GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Allow TensorFlow to dynamically allocate GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ Using GPU: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ No GPU detected, running on CPU")

# ---------------- Data Preprocessing ----------------
train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)

base_dir = r"E:\Emotion_detection_with_CNN\Emotion_detection_with_CNN-main\data"

train_generator = train_data_gen.flow_from_directory(
        os.path.join(base_dir, "train"),
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

validation_generator = validation_data_gen.flow_from_directory(
        os.path.join(base_dir, "test"),
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

# ---------------- Model ----------------
emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

cv2.ocl.setUseOpenCL(False)

emotion_model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(learning_rate=0.0001),
                      metrics=['accuracy'])

# ---------------- Training ----------------
emotion_model_info = emotion_model.fit(
        train_generator,
        steps_per_epoch=28709 // 64,
        epochs=1,
        validation_data=validation_generator,
        validation_steps=7178 // 64)

# ---------------- Save Model ----------------
os.makedirs("Emotion_detection_with_CNN-main/model", exist_ok=True)  # ensure folder exists

with open("Emotion_detection_with_CNN-main/model/emotion_model.json", "w") as json_file:
    json_file.write(emotion_model.to_json())

emotion_model.save_weights("Emotion_detection_with_CNN-main/model/emotion_model.weights.h5")
print("✅ Model and weights saved in 'model/' folder")
