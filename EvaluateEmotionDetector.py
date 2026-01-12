import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Emotion label mapping
emotion_dict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised"
}

# ---------------- Build the model (must match TrainEmotionDetector.py) ----------------
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

# ---------------- Load weights ----------------
emotion_model.load_weights("Emotion_detection_with_CNN-main/model/emotion_model.weights.h5")
print("✅ Loaded weights into model")

# ---------------- Test data generator ----------------
test_data_gen = ImageDataGenerator(rescale=1.0/255)

test_generator = test_data_gen.flow_from_directory(
    'Emotion_detection_with_CNN-main/data/test',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical',
    shuffle=False
)

# ---------------- Prediction ----------------
predictions = emotion_model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

# ---------------- Confusion Matrix ----------------
print("-----------------------------------------------------------------")
c_matrix = confusion_matrix(y_true, y_pred)
print(c_matrix)

cm_display = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=list(emotion_dict.values()))
cm_display.plot(cmap=plt.cm.Blues)
#plt.show()
plt.savefig("Emotion_detection_with_CNN-main/confusion_matrix.png")  # saves the plot instead of blocking
plt.close()

# ---------------- Classification Report ----------------
print("-----------------------------------------------------------------")
print(classification_report(y_true, y_pred, target_names=list(emotion_dict.values())))
