from yolov3.detect_image import YoloV3
import cv2

image = cv2.imread('image-000674.png')
yolov3 = YoloV3(weights_file='weights/middle-20200113-11-55.pt', cfg='cfg/yolov3-custom.cfg')

yolov3.predict([image])