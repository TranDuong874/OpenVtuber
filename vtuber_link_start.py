# coding: utf-8

import numpy as np
import service
import cv2
import sys
import socketio
import base64
import os
from threading import Thread, Event
from queue import Queue
import json

from threading import Thread
from queue import Queue


cap = cv2.VideoCapture(sys.argv[1])
# cap = cv2.VideoCapture(0)

# Get the FPS of the input video
fps = cap.get(cv2.CAP_PROP_FPS)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

model_control = {
    "data" : []
}


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
stop_event = Event()

# ======================================================


def face_detection():
    while not stop_event.is_set():
        ret, frame = cap.read()

        if not ret:
            break

        #   Step 2
        face_boxes, _ = fd.inference(frame) #   Một list các face boxes
        box_queue.put((frame, face_boxes))  #   Frame và các frame boxes của nó
    box_queue.put(None) #   Use none as sentinel value

def face_alignment():
    while True:
        item = box_queue.get()
        if item is None:
            break
        #Step 4
        frame, boxes = item #    Load các frame và boxes ở trong hàng đợi
        landmarks = fa.get_landmarks(frame, boxes) #    Tính toán các landmark dựa vào tham số frame và boxes
        landmark_queue.put((frame, landmarks)) #    Cho frame và landmark tương ứng vào queue
    landmark_queue.put(None) #  Using none as sentinel value

#   Step 7
def iris_localization(YAW_THD=45):
    sio = socketio.Client()
    sio.connect("http://127.0.0.1:6789", namespaces='/kizuna')  #   Connect đến server frontend (Model)

    while True:
        item = landmark_queue.get()
        if item is None:
            break
        frame, preds = item #   Lấy frame và prediction tính được từ landmark queue

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
            
            model_control['data'].append(result_string)
            
            sio.emit('result_data', result_string, namespace='/kizuna')
            upstream_queue.put((frame, landmarks, euler_angle))
            break
        
    sio.disconnect()
    upstream_queue.put(None)



def draw(color=(125, 255, 0), thickness=2):
    sio = socketio.Client()
    sio.connect("http://127.0.0.1:6789", namespaces='/kizuna')
    
    output_dir = 'NodeServer/output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fourcc = cv2.VideoWriter_fourcc(*'H264')
    output_path = os.path.join(output_dir, 'output.mp4')
    out = cv2.VideoWriter(output_path, fourcc, fps, (960, 720))
    
    cnt = 0
    while True:
        item = upstream_queue.get()
        if item is None:
            break
        frame, landmarks, euler_angle = item

        for p in np.round(landmarks).astype(np.int64):
            cv2.circle(frame, tuple(p), 1, color, thickness, cv2.LINE_AA)

        face_center = np.mean(landmarks, axis=0)
        hp.draw_axis(frame, euler_angle, face_center)

        frame = cv2.resize(frame, (960, 720))
        cnt += 1
        out.write(frame)
        
        print(fps)
        #enconde a frame
        # retval, buffer = cv2.imencode('.jpg', frame)
        # jpg_as_text = base64.b64encode(buffer)
    
        # sio.emit('input_data', jpg_as_text, namespace='/kizuna')
        
        
        cv2.imshow('result', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            sio.disconnect()
            break
        
    sio.disconnect()
    print('Frame count: ', cnt)
    cv2.destroyAllWindows()




#   Step 4
alignment_thread = Thread(target=face_alignment)
alignment_thread.start()

#   Step 7
iris_thread = Thread(target=iris_localization)
iris_thread.start()

draw_thread = Thread(target=draw)
draw_thread.start()

face_detection() #  Step 1

alignment_thread.join()
iris_thread.join()
draw_thread.join()

alignment_thread.join()
iris_thread.join()
draw_thread.join()

save_file = open('NodeServer/output/result_data.json', 'w')
json.dump(model_control, save_file, indent=4)
save_file.close()

print('Param count: ', len(model_control['data']))
cap.release()
print('Done')
