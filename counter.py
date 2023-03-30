import cv2
import torch
from tracker import *
import numpy as np
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap=cv2.VideoCapture('cctv.mp4')


def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('IMAGE_FRAME')
cv2.setMouseCallback('IMAGE_FRAME', POINTS)

tracker = Tracker()
area_01=[(377,315),(429,373),(535,339),(500,296)]
area01=set()
while True:
    ret,frame=cap.read()
    frame=cv2.resize(frame,(1020,500))
    cv2.polylines(frame,[np.array(area_01,np.int32)],True,(0,255,0),3)

    results=model(frame)
    # frame=np.squeeze(results.render())
    # a=results.pandas().xyxy[0]
    # print(a)
    list=[]
    for index,row in results.pandas().xyxy[0].iterrows():
        # print(row)
        x1=int(row['xmin'])
        y1=int(row['ymin'])
        x2=int(row['xmax'])
        y2=int(row['ymax'])
        b=str(row['name'])
        if 'person' in b:
            list.append([x1,y1,x2,y2])

        # cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),2)
        # cv2.putText(frame,b,(x1,y1),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
        

    boxes_ids=tracker.update(list)
    # print(boxes_ids)
    for box_id in boxes_ids:
        x,y,w,h,id=box_id
        cv2.rectangle(frame,(x,y),(w,h),(255,0,255),2)
        cv2.putText(frame,str(id),(x,y),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),2)
        result=cv2.pointPolygonTest(np.array(area_01,np.int32),(int(w),int(h)),False)
        print(f'The results are {result}')
        if result>0:
            area01.add(id)
    # print(boxes_ids)
    print(area01)
    print(len(area01))
    p=len(area01)
    cv2.putText(frame,str(p),(20,30),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),2)
    cv2.imshow('Image_FRAME',frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()
    
    
