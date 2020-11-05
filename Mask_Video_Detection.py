import cv2  
import dlib 
import pyttsx3
import playsound

engine = pyttsx3.init()
# =============================================================================
# engine.say('wear a mask')
# engine.runandwait()
# =============================================================================
cap = cv2.VideoCapture(0) 
detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
  
while True: 
    _, frame = cap.read() 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray) 
    for face in faces: 
        x1 = face.left()
        y1 = face.top() 
        x2 = face.right() 
        y2 = face.bottom() 
        p = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        landmarks = predictor(gray, face)
        
        for n in range(31, 36):
            x = landmarks.part(n).x 
            y = landmarks.part(n).y 
            #cv2.rectangle(frame, (x, y), (x+5, y+5), (0, 255, 255))
            t = cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)
            if t.any()==True:
              engine.say('wear a mask')
              engine.runAndWait()
              break

#                playsound.playsound('audio.mp3')
#                break
            print('wear mask')
    cv2.imshow("Frame", frame) 
    key = cv2.waitKey(1)
    if key == 27: 
        break
