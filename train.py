import os
import numpy as np
import cv2 as cv

#this is the code to train the model to recognize the faces listed below


haar_cascade=cv.CascadeClassifier('H:\VSCode\OpenCV\haarcascade_frontalface.xml')

people=['Barack Obama', 'Bill Gates', 'Beyonce']

#storing the path of the base folder in which the 3 image folders are located
DIR = r'H:\VSCode\OpenCV\Projects\Face_Detection_and_Recognition\train'

#training set
features=[]
labels=[]

#creating a funcrion that loops over every folder in the base folder, and in every image folder, loops over every image and grabs the face in that image, and adds it to the training set
def create_train():
    for person in people:
        path=os.path.join(DIR, person) #this gives the path of every image folder located inside the base folder
        label=people.index(person) #index of every item in the people list

        for img in os.listdir(path): #looping through every image in the image folder
            img_path=os.path.join(path, img) #obtaining the path of every image

            img_array=cv.imread(img_path) #reading every image
            gray=cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect=haar_cascade.detectMultiScale(gray, 1.1, 5) #detecting each face in the grayscale image

            for(x, y, w, h) in faces_rect:
                face_roi= gray[y:y+h, x:x+w] #cropping out only the face from the image
                features.append(face_roi)
                labels.append(label)

create_train()

#conevrting the features and labels list into a numpy array
features=np.array(features, dtype='object')
labels=np.array(labels)

print(f'the number of faces detected in total in the base directory={len(features)}')

#we can now use our features list and the labels list to train our recognizer

face_recognizer=cv.face.LBPHFaceRecognizer_create() #instatiates the face recognizer

face_recognizer.train(features, labels)

#np.save() function is used to store the input array in a disk file with npy extension(.npy).
np.save('features.npy', features)
np.save('labels.npy', labels)

face_recognizer.save('face_train.yml')

