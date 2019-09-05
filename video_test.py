from face_recog import *
import cv2

known_encodings, known_names = get_known_encodings('faces')
print(known_names[0])
print(known_encodings[0])

cap = cv2.VideoCapture(0)

while True:

    _, frame = cap.read()

    try:
        boxes = get_boxes(frame)

        faces = get_faces(frame)

        encodings = get_embedding_list(faces)

        names = compare_faces(encodings, known_encodings, known_names)

        boxes_and_names(frame, boxes, names)

    except:
        pass

    cv2.imshow('recog', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()