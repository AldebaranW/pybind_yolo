from ultralytics import YOLO
import cv2
import numpy as np
import _process
# TODO: 神经网络在识别矿石上效果较好，但存在将不是矿石的物体识别成矿石的问题（可以利用置信度判断>0.9大致解决）

model_name = '/home/wu/GitHub/ultralytics/runs/segment/train14/weights/best.pt' # 模型位置
img = cv2.imread("/home/wu/GitHub/datasets/MineData/images/1735975.jpg") # 识别的图像位置
model = YOLO(model=model_name)
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, img =  cap.read()
    res = model(img, device='cpu')
    if res[0].boxes != None and res[0].masks != None:
        box = res[0].boxes.boxes.numpy()
        mask = res[0].masks.masks.numpy()

        for i in range(box.shape[0]):
            # print(type(box.tolist()), type(mask.tolist()))
            box_ = box.tolist()[i]
            if box_[-2] < 0.9:
                continue
            mask_ = mask[i].reshape(640, 640, -1)
            mask_ = cv2.Mat(mask_)
            _, mask_ = cv2.threshold(mask_, 0.5, 1, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(np.uint8(mask_), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, (255, 255, 0), 3)
            # cv2.imshow('img', img)
            # cv2.waitKey(1000)
            contours = contours[0].tolist()
            contours = [i[0] for i in contours]
            _process.dataprocess(box_, contours)