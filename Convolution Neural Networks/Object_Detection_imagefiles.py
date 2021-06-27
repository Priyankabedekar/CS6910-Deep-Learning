import cv2
import numpy as np

net = cv2.dnn.readNet('yolov3.weights','yolov3.cfg')
classes = []
with open('coco.names','r') as f:
    classes = f.read().splitlines()

img = cv2.imread('cctv.png')
height, width, _ = img.shape

'''Input image after mean substraction, normalising, channel swapping, r-0, g-1, b-2, inputs to our model'''
# for b in blob:
#     for n, img_blob in enumerate(b):
#         cv2.imshow(str(n), img_blob)


#rescale, size of image, mean subtraction.
blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), (0,0,0), swapRB = True, crop = False) 
#Set Input from blob to network.
net.setInput(blob)
#get output layers names
output_layers_names = net.getUnconnectedOutLayersNames()
#Pass names to forward pass, run FP and output at each layer.
layerOutputs = net.forward(output_layers_names)

boxes = [] #Extract bounding boxes.
confidences = []
class_ids = [] #80

#85 parameters

#Loop over layers output
for output in layerOutputs:
    for detection in output:
        scores = detection[5:]            #all 80 class prediction
        class_id = np.argmax(scores)      # max predicted class
        confidence = scores[class_id]     # Class with highest confidence
        if confidence > 0.5:
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)

            x = int(center_x - w/2)
            y = int(center_y - h/2)

            boxes.append([x,y,w,h])
            confidences.append((float(confidence)))
            class_ids.append(class_id)


indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0,255, size = (len(boxes),3))

if len(indexes)>0:
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i],2))
        color = colors[i]
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)


cv2.imshow('cctv.png', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
