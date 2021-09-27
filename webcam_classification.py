import numpy as np
import os
import cv2
from tensorflow.keras.models import load_model


train_path="E:/Downloads/casting_product_image/casting_data/casting_data/train/"
test_path="E:/Downloads/casting_product_image/casting_data/casting_data/test/"



classes=os.listdir(train_path)
print((classes))
no_of_classes=len(classes)


model1=load_model('trained_model1.h5')

width=640
height=480
threshold=0.7

cap=cv2.VideoCapture(0)
cap.set(3,width)
cap.set(4,height)

def pre_processing(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)
    img=img/255
    return img

while True:
    succes,img_original=cap.read()
    img=np.asarray(img_original)
    img=cv2.resize(img,(50,50))
    img=pre_processing(img)
    img=img.reshape(1,50,50,1)
    class_index=int(model1.predict_classes(img))
    #print(class_index)
    prediction=model1.predict(img)
    probVal=np.amax(prediction)
    if probVal>0.8:
        #print(class_index,probVal)
        cv2.putText(img_original,classes[class_index]+" "+str(probVal),(50,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    cv2.imshow("out",img_original)

    if cv2.waitKey(1) & 0XFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
