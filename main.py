import threading

import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

counter = 0

face_match = False
#here you put a photo of your face to compare it
#you have to save the foto in the face directory where you have your program
reference_img = cv2.imread("")

def check_face(frame):
    global face_match
    try:
        if DeepFace.verify(frame,reference_img.copy())['verified']:
            face_match = True
        else:
            face_match = False
    except ValueError:
        face_match = False


while True:
    ret , frame = cap.read()

    if ret:
        if counter %30 ==0:
            try:
                threading.Thread(target=check_face,args=(frame.copy(),)).start()
            except ValueError:
                pass
        counter += 1

        if face_match:
            cv2.putText(frame,"Match",(20,450),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),thickness=3)
        else:
            cv2.putText(frame, "No Match", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=3)

        cv2.imshow("Video",frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyWindow()

