import os

path = r"E:\ Emotion_detection_with_CNN\data\ train"

print("Does the path exist? ", os.path.exists(path))
if os.path.exists(path):
    print("Subfolders:", os.listdir(path))
else:
    print("❌ Path not found!")
