import cv2
import numpy as np

# Load the model
# Download these files from OpenCV's GitHub repo or somewhere else
model = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', model) 

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    
    if not ret:
        continue
        
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104.0, 177.0, 123.0]) # Preprocessing for MobileNet
    
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]): # For each detected face
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5: # Filter out weak detections by a threshold of 0.5
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Display the bounding box of faces
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
    
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to quit the program
        break
        
cap.release()
cv2.destroyAllWindows()
