import cv2
from matplotlib import pyplot as plt
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
cas = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

image_frame = []
while cap.isOpened():
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray)
    for (x, y, w, h) in detected_faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        face_frame_gray = gray[y:y+h, x:x+w]
        face_frame_img = img[y:y + h, x:x + w]
        eyes = cas.detectMultiScale(face_frame_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face_frame_img, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

    image_frame.append(img)
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



