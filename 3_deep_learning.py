import cv2
import numpy as np

# Load the model
# Download these files from OpenCV's GitHub repo or somewhere else
threshold_conf = 0.7
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

    # How many faces?
    num_faces = 0

    for i in range(detections.shape[2]): # For each detected face
        confidence = detections[0, 0, i, 2]
        if confidence > threshold_conf: # Filter out weak detections by a threshold of 0.5
            num_faces = num_faces + 1
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Display the bounding box of faces
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
    
    cv2.imshow('Video', frame)

    if num_faces > 1:
        cv2.namedWindow('2 faces!', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('2 faces!', cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        cv2.imshow('2 faces!', frame)
        cv2.waitKey(3000000)
        break 
    
    if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to quit the program
        break
        
cap.release()
cv2.destroyAllWindows()
