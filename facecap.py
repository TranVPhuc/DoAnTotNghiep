from tkinter import messagebox
import threading
import cv2
import os
import time

def Face_Cap(name, face_id):
    
    font = cv2.FONT_HERSHEY_COMPLEX
    
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video width
    cam.set(4, 480) # set video height

    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    print("\n [INFO] Initializing face capture. Look the camera and wait ...")
    # Initialize individual sampling face count
    
    count = 0
    def save_picture(face,count):         
        for (x,y,w,h) in faces:          
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)        
            if not os.path.exists("dataset"):
                os.makedirs("dataset")
            if not os.path.exists(f"dataset/{name}"):
                os.makedirs(f"dataset/{name}")
            # Save the captured image into the datasets folder
            cv2.imwrite(f"dataset/{name}/"+ str(name)+ '.' + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
               
    while(True):
        ret, img = cam.read()
        img = cv2.flip(img, 1) # flip video image vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5) 
        cv2.putText(img, str("Dang lay du lieu vui long cho!"), (0, 20), font, 1, (255, 255, 255), 1)   
        if count <= 300:
            t1 = threading.Thread(target=save_picture,args=[faces,count])
            t1.start()
            t1.join() 
            count +=1
            
        else:
            break
        cv2.imshow("Gui_2",img)
        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()
