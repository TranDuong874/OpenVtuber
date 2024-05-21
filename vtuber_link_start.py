# coding: utf-8

import numpy as np
import service
import cv2
import sys
import socketio

from threading import Thread
from queue import Queue


cap = cv2.VideoCapture(sys.argv[1])

fd = service.UltraLightFaceDetecion("weights/RFB-320.tflite",
                                    conf_threshold=0.98)
fa = service.CoordinateAlignmentModel("weights/coor_2d106.tflite")
hp = service.HeadPoseEstimator("weights/head_pose_object_points.npy",
                               cap.get(3), cap.get(4))
gs = service.IrisLocalizationModel("weights/iris_localization.tflite")

QUEUE_BUFFER_SIZE = 18

box_queue = Queue(maxsize=QUEUE_BUFFER_SIZE)
landmark_queue = Queue(maxsize=QUEUE_BUFFER_SIZE)
iris_queue = Queue(maxsize=QUEUE_BUFFER_SIZE)
upstream_queue = Queue(maxsize=QUEUE_BUFFER_SIZE)

# ======================================================


def face_detection():
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        #   Step 2
        face_boxes, _ = fd.inference(frame) #   Một list các face boxes
        box_queue.put((frame, face_boxes))  #   Frame và các frame boxes của nó


def face_alignment():
    while True:
        #Step 4
        frame, boxes = box_queue.get() #    Load các frame và boxes ở trong hàng đợi
        landmarks = fa.get_landmarks(frame, boxes) #    Tính toán các landmark dựa vào tham số frame và boxes
        landmark_queue.put((frame, landmarks)) #    Cho frame và landmark tương ứng vào queue


#   Step 7
def iris_localization(YAW_THD=45):
    sio = socketio.Client()

    sio.connect("http://127.0.0.1:6789", namespaces='/kizuna')  #   Connect đến server frontend (Model)

    while True:
        frame, preds = landmark_queue.get() #   Lấy frame và prediction tính được từ landmark queue

        for landmarks in preds: #   Với mỗi danh sách landmark được dự đoán trong predictions
            
            # calculate head pose
            euler_angle = hp.get_head_pose(landmarks).flatten() #   Step 8 lấy headpose
            pitch, yaw, roll = euler_angle 
            # Roll  phi     trục x vuông góc với mặt      (Nghiêng đầu)
            # Pitch theta   trục y song song với mặt      (Cúi/ngẩng đầu)
            # Yaw   psi     trụ  z vuông góc mặt đỉnh đầu (Xoay trái phải)   

            eye_starts = landmarks[[35, 89]] # [mắt trái, mắt phải]
            eye_ends = landmarks[[39, 93]] 
            eye_centers = landmarks[[34, 88]]
            eye_lengths = (eye_ends - eye_starts)[:, 0]
            

            pupils = eye_centers.copy() #   Lấy landmark trung tâm của con ngươi
            
            #   Step 8
            if yaw > -YAW_THD:
                #   Lấy mesh của lông mày trái
                #   Landmarks của iris
                iris_left = gs.get_mesh(frame, eye_lengths[0], eye_centers[0]) 
                pupils[0] = iris_left[0]

            #   Xử lý exception quay đầu che mất một mắt?
            if yaw < YAW_THD:
                iris_right = gs.get_mesh(frame, eye_lengths[1], eye_centers[1])
                pupils[1] = iris_right[0]

            #   Step 9
            poi = eye_starts, eye_ends, pupils, eye_centers
            
            theta, pha, _ = gs.calculate_3d_gaze(poi) # Tính góc gì đó?  
            
            #????????????????????????????????????????????????????
            mouth_open_percent = (
                landmarks[60, 1] - landmarks[62, 1]) / (landmarks[53, 1] - landmarks[71, 1])
            left_eye_status = (
                landmarks[33, 1] - landmarks[40, 1]) / eye_lengths[0]
            right_eye_status = (
                landmarks[87, 1] - landmarks[94, 1]) / eye_lengths[1]
            
            #????????????????????????????????????????????????????
            result_string = {'euler': (pitch, -yaw, -roll),
                             'eye': (theta.mean(), pha.mean()),
                             'mouth': mouth_open_percent,
                             'blink': (left_eye_status, right_eye_status)}
            sio.emit('result_data', result_string, namespace='/kizuna')
            upstream_queue.put((frame, landmarks, euler_angle))
            break


def draw(color=(125, 255, 0), thickness=2):
    while True:
        frame, landmarks, euler_angle = upstream_queue.get()

        for p in np.round(landmarks).astype(np.int64):
            cv2.circle(frame, tuple(p), 1, color, thickness, cv2.LINE_AA)

        face_center = np.mean(landmarks, axis=0)
        hp.draw_axis(frame, euler_angle, face_center)

        frame = cv2.resize(frame, (960, 720))

        cv2.imshow('result', frame)
        cv2.waitKey(1)




#   Step 4
alignment_thread = Thread(target=face_alignment)
alignment_thread.start()

#   Step 7
iris_thread = Thread(target=iris_localization)
iris_thread.start()

draw_thread = Thread(target=draw)
draw_thread.start()

face_detection() #  Step 1

cap.release()
cv2.destroyAllWindows()
