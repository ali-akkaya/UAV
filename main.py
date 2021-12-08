

import cv2
from time import sleep
import numpy as np
#camera that is plugged in
def drawBox(img,bbox):
    x,y,w,h = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),3,1)

def main():
    CAMERA_PORT = 0
    
    camera = cv2.VideoCapture(CAMERA_PORT,cv2.CAP_DSHOW)
    
    for i in range(0,20):
        _,img = camera.read()
    
    boundingBox = cv2.selectROI("camera",img,False)
    
    
    
    
    tracker_mosse = cv2.TrackerMOSSE_create()
    tracker_csrt = cv2.TrackerCSRT_create()
    tracker_kcf= cv2.TrackerKCF_create()
    tracker_MedianF = cv2.TrackerMedianFlow_create()
    tracker_mil = cv2.TrackerMIL_create()
    tracker_tld = cv2.TrackerTLD_create()
    tracker_boosting = cv2.TrackerBoosting_create()
    
    tracker_mosse.init(img,boundingBox)
    tracker_csrt.init(img,boundingBox)
    tracker_kcf.init(img,boundingBox)
    tracker_MedianF.init(img,boundingBox)
    tracker_mil.init(img,boundingBox)
    tracker_tld.init(img,boundingBox)
    tracker_boosting.init(img,boundingBox)
    
    
    
    while True:
        timer = cv2.getTickCount()
        
        
        success, img = camera.read()
        img_mosse = img.copy()
        img_csrt = img.copy()
        img_kcf = img.copy()
        img_MedianF = img.copy()
        img_mil = img.copy()
        img_tld = img.copy()
        img_boosting = img.copy()
        success_mosse, boundingBox_mosse = tracker_mosse.update(img)
        success_csrt, boundingBox_csrt = tracker_csrt.update(img)
        success_kcf, boundingBox_kcf = tracker_kcf.update(img)
        success_MedianF, boundingBox_MedianF = tracker_MedianF.update(img)
        success_mil, boundingBox_mil = tracker_mil.update(img)
        success_tld, boundingBox_tld = tracker_tld.update(img)
        success_boosting, boundingBox_boosting = tracker_boosting.update(img)
    
        if success:
            if success_mosse:
                drawBox(img_mosse,boundingBox_mosse)
                cv2.putText(img_mosse,"Success: True-Type:Mosse", (75,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
            else:                
                cv2.putText(img_mosse,"Success: False-Type:Mosse", (75,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

            if success_csrt:
                drawBox(img_csrt,boundingBox_csrt)
                cv2.putText(img_csrt,"Success: True-Type:csrt", (75,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
            else:
                cv2.putText(img_csrt,"Success: False-Type:csrt", (75,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
            
            if success_kcf:
                drawBox(img_kcf,boundingBox_kcf)
                cv2.putText(img_kcf,"Success: True-Type:KCF", (75,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
            else:                
                cv2.putText(img_kcf,"Success: False-Type:KCF", (75,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
            
            if success_MedianF:
                drawBox(img_MedianF,boundingBox_MedianF)
                cv2.putText(img_MedianF,"Success: True-Type:MedianF", (75,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
            else:                
                cv2.putText(img_MedianF,"Success: False-Type:MedianF", (75,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

            if success_mil:
                drawBox(img_mil,boundingBox_mil)
                cv2.putText(img_mil,"Success: True-Type:MIL", (75,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
            else:                
                cv2.putText(img_mil,"Success: False-Type:MIL", (75,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

            if success_tld:
                drawBox(img_tld,boundingBox_tld)
                cv2.putText(img_tld,"Success: True-Type:TLD", (75,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)     
            else:                
                cv2.putText(img_tld ,"Success: False-Type:TLD", (75,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

            if success_boosting:
                drawBox(img_boosting,boundingBox_boosting)
                cv2.putText(img_boosting,"Success: True-Type:Boosting", (75,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)      
        else: 
            cv2.putText(img_boosting,"Success: False-Type:Boosting", (75,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
            cv2.putText(img_csrt,"Success: False-Type:csrt", (75,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
            cv2.putText(img_tld,"Success: False-Type:TLD", (75,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)     
            cv2.putText(img_mil,"Success: False-Type:MIL", (75,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
            cv2.putText(img_MedianF,"Success: False-Type:MedianF", (75,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
            cv2.putText(img_kcf,"Success: False-Type:KCF", (75,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
            cv2.putText(img_mosse,"Success: False-Type:Mosse", (75,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)


        fps = cv2.getTickFrequency() / (cv2.getTickCount()-timer)
            
        cv2.putText(img,"fps:" + str(int(fps)), (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255),2)
       
        cv2.imshow("tracker_mosse", img_mosse)
        cv2.imshow("tracker_csrt", img_csrt)
        cv2.imshow("tracker_kcf", img_kcf)
        cv2.imshow("tracker_MedianF", img_MedianF)
        cv2.imshow("tracker_mil", img_mil)
        cv2.imshow("tracker_tld", img_tld)
        #cv2.imshow("tracker_boosting", img_boosting)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break;
            
            
            
    cv2.destroyAllWindows()        
main()    
