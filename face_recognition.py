import cv2 as cv
import numpy as np

#this is the code that performs the actual face recognition on the sample image provided by the user

haar_cascade=cv.CascadeClassifier('H:\VSCode\OpenCV\haarcascade_frontalface.xml')

people=['Bill Gates','Barack Obama', 'Beyonce']

#loading the features and labels numpy arrays that we saved into the disc earlier
#features=np.load('features.npy', allow_pickle=True)
#labels=np.load('labels.npy')

face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_train.yml') #reading our trained model

#reading our test image to be recognized
face_tobe_tested=cv.imread(r'H:\VSCode\OpenCV\Projects\Face_Detection_and_Recognition\test\billgates.jpg')
face_tobe_tested=cv.resize(face_tobe_tested, (0, 0), fx=1.0, fy=1.0)

#now converting this image to grayscale
gray=cv.cvtColor(face_tobe_tested, cv.COLOR_BGR2GRAY)

#detect the face in the image
face_rect=haar_cascade.detectMultiScale(gray, 1.1, 7)

for (x, y, w, h) in face_rect:
    faces_roi=gray[y:y+h, x:x+w]

    #now we can predict using the built in face recognizer

    label, confidence=face_recognizer.predict(faces_roi)

    if(confidence>100):
        print("unknown")
        cv.putText(face_tobe_tested, "Unknown", (60, 150), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
    else:
        print(f'Label={people[label]} with a confidence of {confidence}')
        cv.putText(face_tobe_tested, str(people[label]), (60, 150), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)

    cv.rectangle(face_tobe_tested, (x,y), (x+w, y+h), (255, 0, 0), 3)


cv.imshow("Recognized face", face_tobe_tested)

cv.waitKey(0)