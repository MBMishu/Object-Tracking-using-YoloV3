import numpy as np
import time
import cv2
import os
from sort import *


tracker = Sort()
memory = {}
line = [(0, 180), (700, 180)]
counter = 0

cap = cv2.VideoCapture('sample_3.mp4')


(W, H) = (None, None)

whT = 320
confThreshold  = 0.5 # Threshold to detect object
nmsThreshold= 0.4

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Name custom object
classes = ["car"]

while True:
    success,frame = cap.read()
    
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    layerOutputs = net.forward(output_layers)
    
    hT, wT, cT = frame.shape
    if W is None or H is None:
    	(H, W) = frame.shape[:2]
     
    bbox = []
    confs = []
    classIDs = []
    
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            if confidence > confThreshold:
                box = detection[0:4] * np.array([W, H, W, H])
                
                (centerX, centerY, width, height) = box.astype("int")
    
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                
                bbox.append([x, y, int(width), int(height)])
                confs.append(float(confidence))
                classIDs.append(classID)
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    
    dets = []
    
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (bbox[i][0], bbox[i][1])
			
            (w, h) = (bbox[i][2], bbox[i][3])
			
            dets.append([x, y, x+w, y+h, confs[i]])
    
    # np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    dets = np.asarray(dets)
    tracks = tracker.update(dets)
    
    
    bbox = []
    indexIDs = []
    previous = memory.copy()
    memory = {}
    
    for track in tracks:
        bbox.append([track[0], track[1], track[2], track[3]])
        indexIDs.append(int(track[4]))
        memory[indexIDs[-1]] = bbox[-1]
        
    if len(bbox) > 0:
        i = int(0)
        for box in bbox:
            (x, y) = (int(box[0]), int(box[1]))
            (w, h) = (int(box[2]), int(box[3]))
            
            color = (128,0,128)
            
            cv2.rectangle(frame, (x, y), (w, h), color, 2)
            
            if indexIDs[i] in previous:
                
                previous_box = previous[indexIDs[i]]
                (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                cv2.line(frame, p0, p1, color, 3)
                
                if intersect(p0, p1, line[0], line[1]):
                    counter += 1
                
            text = "{}".format(indexIDs[i])
            cv2.putText(frame, text, (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
            i += 1
                
                
 
    
    # draw line
    cv2.line(frame, line[0], line[1], (0, 255, 255), 5)

    # draw counter
    cv2.putText(frame,str(counter),
                  (580, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(frame,"Masum Billah",
                  (280, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    # counter += 1

    cv2.imshow('JunkYard',frame)

    if cv2.waitKey(20) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()