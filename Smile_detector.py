import cv2

cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
cascade_eye = cv2.CascadeClassifier('haarcascade_eye.xml') 
cascade_smile = cv2.CascadeClassifier('haarcascade_smile.xml')

vc = cv2.VideoCapture(0)\

def TakeSnapshotAndSave():
    num = 0
    while num < 2: # Make 2 snapshot 
        _, image = vc.read()
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face = cascade_face.detectMultiScale(grayscale, 1.3, 5)
        for (x_face, y_face, w_face, h_face) in face:
            cv2.rectangle(image, (x_face, y_face), (x_face + w_face, y_face + h_face), (255, 130, 0), 
                         2)
            ri_gray_scale = grayscale[y_face:y_face + h_face, x_face:x_face + w_face]
            ri_color = image[y_face:y_face + h_face, x_face:x_face + w_face]
            eye = cascade_eye.detectMultiScale(ri_gray_scale, 1.3, 18) #Depending from hyperparameters model can detect 
            #close eyes as opened, but after playing with neughbours this problem can be solved. In this case I used optimal hyperparameters 
            for (x_eye, y_eye, w_eye, h_eye) in eye:
                cv2.rectangle(ri_color, (x_eye, y_eye), (x_eye + w_eye, y_eye + h_eye),
                             (0, 180, 60), 2)
                smile = cascade_smile.detectMultiScale(ri_gray_scale, 1.7, 20)
                for (x_smile, y_smile, w_smile, h_smile) in smile:
                    cv2.rectangle(ri_color, (x_smile, y_smile), (x_smile + w_smile, y_smile + h_smile),
                                (0, 180, 60), 2)
                    x = 0
                    y = 20
                    text_color = (0,255,0)
                    cv2.imwrite('opencv'+str(num)+'.jpg',image)
                    num = num+1

            
        cv2.imshow('Video', image)
        if cv2.waitKey(1) & 0xFF == ord('w'):
            break
if __name__ == "__main__":
    TakeSnapshotAndSave()


vc.release() 
cv2.destroyAllWindows() 