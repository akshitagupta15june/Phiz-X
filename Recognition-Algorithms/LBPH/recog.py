import numpy as np
import cv2
import os

data_path = '/home/abhishek/Documents/Face/'


def face_detection(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face = face_classifier.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    if face is ():
        return img, []
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi = gray[y:y + h, x:x + w]
    return img, roi


def prepare_data(path):
    labels = []
    faces = []
    for i, images in enumerate(os.listdir(path)):
        #print(images)
        img_path = path + images
        face = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        faces.append(np.asarray(face, dtype=np.uint8))
        labels.append(i)
    #labels.append(np.asarray(labels, dtype=np.int32))
    return faces, labels


Faces, Labels = prepare_data(data_path)

model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Faces), np.asarray(Labels))
print("Model Training Completed")

def image_prediction(img):

    result = model.predict(img)
    pr = result[1]/500;
    prob = (1 - pr)*100;
    return prob;



cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    frame, face = face_detection(frame)
    try:
        prob = image_prediction(face)
        if int(prob) > 80:
            cv2.putText(frame, str(prob)+"% Same User", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, str(prob) + "Not a same User", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    except:
        cv2.putText(frame, "Face not Found", (100, 200),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('img', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
