from ultralytics import YOLO
import cv2

from sort.sort import *
from util import get_receipt, read_total, write_csv

results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('yolov8n.pt')
total_detector = YOLO('./models/total_detector.pt')

# load video
cap = cv2.VideoCapture('./sample.mp4')

# <----------------------------------------------------------------
'''
receipt_class = 2
receipt_classes = [receipt_class]
'''
# ---------------------------------------------------------------->

# read frames
frame_number = -1
ret = True

while ret:
    frame_number += 1
    ret, frame = cap.read()
    if ret:
        #results[frame_number] = {}
        # detect receipts <----------------------------------------
        '''
        detections = coco_model(frame)[0]
        detections_ids = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection 
            if int(class_id) in paper:
                detections_ids.append([x1, y1, x2, y2, score])

        # track vehicles

        track_ids = mot_tracker.update(np.asarray(detections_ids))      
        '''
        # detect receipts ----------------------------------------->

        totals = total_detector(frame)[0]
        for total in totals.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = total 

            # assign total to receipt <-----------------------------
            '''
            xrec1, yrec1, xrec2, yrec2, rec_id = get_receipt(total, track_ids)

            if rec_id != -1
            '''
            # crop total ------------------------------------------->

            total_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

            # process total

            total_crop_gray = cv2.cvtColor(total_crop, cv2.COLOR_BGR2GRAY)
            _, total_crop_thresh = cv2.threshold(total_crop_gray, 205, 255, cv2.THRESH_BINARY_INV)

            '''
            cv2.imshow('original_crop', total_crop)
            cv2.imshow('threshold', total_crop_thresh)

            cv2.waitKey(0)
            '''

            # read total number

            total_text, total_text_score = read_total(total_crop_thresh)

            if total is not None:
                '''
                results[frame_number][receipt_id] = {'receipt': {'bbox': [xrec1, yrec2, xrec2, yrec2]},
                                                  'total': {'bbox': [x1, y1, x2, y2],
                                                            'text': [total_text],
                                                            'bbox_score': [score],
                                                            'text_score':[total_text_score]}}
                '''
                results[frame_number] = {'total': {'bbox': [x1, y1, x2, y2],
                                                    'text': [total_text],
                                                    'bbox_score': [score],
                                                    'text_score':[total_text_score]}}

# write results 
write_csv(results, './test.csv')