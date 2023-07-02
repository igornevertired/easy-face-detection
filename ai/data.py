import cv2

vid = cv2.VideoCapture(0)
face = cv2.CascadeClassifier('faces.xml');

while True:
    ret, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detection = face.detectMultiScale(gray, 1.3, 4)

    if(len(detection) > 0):
        (x,y,w,h) = detection[0]
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h), (0,255,0), 2)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
vid.release()
cv2.destroyAllWindows()




