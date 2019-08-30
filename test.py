from face_recog import *
import cv2

known_encodings, known_names = get_known_encodings('faces')
print(known_names[0])
print(known_encodings[0])

cap = cv2.VideoCapture(0)

while True:

    _, frame = cap.read()
    a = get_faces(frame)
    print(a)
    cv2.imshow('vid', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()