#This file is for taking the user's face and outputing results

print("[INFO] Loading Modules")

import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

print("[INFO] Modules Loaded")

print("[INFO] Staring Camera")
# model[0] works fine. Rest all overfits
models = { 0: '../trained_models/fer_0-1426_0-9519.h5',
        1: '../trained_models/fer_0-1042_0-9651.h5',
        2: '../trained_models/fer_0-0268_0-9910.h5'
        }


SHOW_TEXT = False
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

model = tf.keras.models.load_model(models[0])
emotion = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Suprise']
curr_emoji = None

def update_curr_emoji(curr_emoji):
    with open('curr_emoji.txt', 'w') as f:
        f.write(str(curr_emoji))

def disp_text(text,image):
    font = cv2.FONT_HERSHEY_SIMPLEX
    #org = (image.shape[0]-50,image.shape[1]-50)
    org=(50,50)
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    image = cv2.putText(image, text, org, font,fontScale, color, thickness, cv2.LINE_AA)
    return image

def load_emojis(dir):
    img = []
    for i in range(len(emotion)):
        img.append(cv2.imread(f'{dir}/{i}.png'))
    return img


emojis = load_emojis('../emojis')

pred = None
while(True):
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 6)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = cv2.resize(roi_gray,(48,48))
        cropped_img = np.expand_dims(cropped_img, axis=-1)
        pred_img = cropped_img / 255
        pred_img = pred_img
        #print(pred_img.shape)
        prev_pred = pred
        pred = np.argmax(model.predict(np.array([pred_img]))[0])
        if(prev_pred != pred):
            update_curr_emoji(pred)
        if SHOW_TEXT:
            frame = disp_text(emotion[pred],frame)
        try:
            emoji_img = cv2.resize(emojis[pred],(256,256))
            cv2.imshow("Emoji",emoji_img)
        except:
            pass
    frame = cv2.resize(frame,(600,400))
    cv2.imshow('Face Image',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("[INFO] Done....Releasing Recources")
cap.release()
cv2.destroyAllWindows()
