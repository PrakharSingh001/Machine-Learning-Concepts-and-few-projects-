import cv2
video=cv2.VideoCapture(0)
while True:
    ret,img=video.read()
    cv2.imshow('img',img)
    cv2.waitKey(1)

