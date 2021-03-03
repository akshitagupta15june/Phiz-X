import cv2
from matplotlib import pyplot as plt
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
cas = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
smile_cas = cv2.CascadeClassifier('haarcascade_smile.xml')
glasses = cv2.imread('glass.png', -1)
cigar = cv2.imread('cigar.png', -1)

while cap.isOpened():
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in detected_faces:
      
        roi_face = img[y:y+h, x:x+w]
        roi_gray = gray[y:y+h, x:x+w]
        eyes = cas.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)
        for (ex, ey, ew, eh) in eyes:
            
            roi_eyes = roi_face[ey:ey+eh, x:x+w]
            glasses_re = cv2.resize(glasses, (int(w), eh), interpolation=cv2.INTER_CUBIC)
            gw, gh, gc = glasses_re.shape;
            
            for i in range(0, gw):
                for j in range(0, gh):
                    if glasses_re[i, j][3] != 0:
                        roi_face[ey+i, j] = glasses_re[i, j]

        smile = smile_cas.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)
        for (sx, sy, sw, sh) in smile:
         
            roi_smile = roi_face[sy:sy+sh, sx:sx+sw]
            cigar_re = cv2.resize(cigar, (int(sw/1.8), int(sh/1.5)), interpolation=cv2.INTER_CUBIC)
            cw, ch, cc = cigar_re.shape
         
            for i in range(0, cw):
                for j in range(0, ch):
                    if cigar_re[i, j][3] != 0:
                        roi_face[int(1.2*sy)+i, int(1.8*sx)+j] = cigar_re[i, j]

    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
