import cv2
import numpy as np
import os
import access_database as db
import threading

def TEST():
    #connect
    cnx = db.connect_to_db('127.0.0.1','root','vlo136fv',1306,'face_data')
    cursor = cnx.cursor()
    
    #folder_path = os.getcwd()+"\\dataset\\"
    # Tạo một mảng để lưu tên các tệp tin
    #file_names = os.listdir(folder_path)
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # iniciate id counter
    id = 0

    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video widht
    cam.set(4, 480)  # set video height

    # Define min window size to be recognized as a face
    minW = 0.05 * cam.get(3)
    minH = 0.05 * cam.get(4)

    #get list name
    names = db.get_list(cnx.cursor())
        
    def get_name_b_MSV(id,names):
        for i,name,msv in names:
            if i == id:
                return name

    def face_detect(faces):
        for (x, y, w, h) in faces:
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])           
            # Check if confidence is less them 100 ==> "0" is perfect match
            if (confidence < 60):            
                
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                fname = get_name_b_MSV(id,names)
                
            else:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                fname = "unknown"
                
            cv2.putText(img, str(fname), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            

    while True:

        ret, img = cam.read()
        img = cv2.flip(img, 1) # Flip vertically

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=2,
            minSize=(int(minW), int(minH)),
        )
        
        t1 = threading.Thread(target=face_detect,args=[faces])
        t1.start()
        t1.join() 

        cv2.imshow('Chuong trinh nhan dien khuon mat ', img)

        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break
    
    db.close_cnc(cnx)
    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()
    
