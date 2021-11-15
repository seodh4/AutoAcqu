import cv2
import numpy as np
 
# Create a VideoCapture object
cap = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)
# cap2 = cv2.VideoCapture(2)
cap3 = cv2.VideoCapture(3)
# Check if camera opened successfully
if (cap.isOpened() == False): 
  print("Unable to read camera feed")
 
# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.

 
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.

while(True):
    ret, frame = cap.read()
    ret, frame1 = cap1.read()
    # ret, frame2 = cap2.read()
    ret, frame3 = cap3.read()
 
    cv2.imshow('frame',frame)
    cv2.imshow('frame1',frame1)
    # cv2.imshow('frame2',frame2)
    cv2.imshow('frame3',frame3)
    # cv2.imshow('frame453',frame2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

 
# When everything done, release the video capture and video write objects
cap.release()
cap1.release()
# cap2.release()
cap3.release()
# Closes all the frames
cv2.destroyAllWindows() 
