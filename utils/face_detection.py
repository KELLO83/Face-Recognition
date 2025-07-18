

from ultralytics import YOLO
import cv2
import torch
import numpy as np


"""
얼굴탐지 모델 바운딩박스 좌표 반환진행

face detection -> face feature extraction -> face recognition

"""


class face_yolo_detection:
    def __init__(self, model_path='weight/yolov11l-face.pt'):
        self.model = YOLO(model_path).to('cuda:0' if torch.cuda.is_available() else 'cpu')

    def detect_faces(self, image) -> list:
        results = self.model(image, conf=0.5, iou=0.5)
        face_detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # 박스 좌표
            scores = result.boxes.conf.cpu().numpy()  # 신뢰도 점수
            class_ids = result.boxes.cls.cpu().numpy()  # 클래스 ID

            for box, score, class_id in zip(boxes, scores, class_ids):
                x1, y1, x2, y2 = map(int, box)
                face_detections.append({
                    "box": (x1, y1, x2, y2),
                    "score": score,
                    "class_id": class_id
                })
        return face_detections
    


if __name__ == '__main__':
    sample_image = cv2.imread('test_dataset/0033_wider_indoor_scene0326_00000071.jpg')
    detector = face_yolo_detection()
    detections = detector.detect_faces(sample_image) # 반환값 ({"box": (x1, y1, x2, y2), "score": score, "class_id": class_id})

    cv2.namedWindow('f', cv2.WINDOW_NORMAL)
    for i, detection in enumerate(detections):
        box = detection['box']
        score = detection['score']
        x1, y1, x2, y2 = box
        cv2.rectangle(sample_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(sample_image, f'Score: {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        focus_area = sample_image[y1:y2, x1:x2]
        cv2.namedWindow(f'focus_area_{i}', cv2.WINDOW_NORMAL)
        cv2.imshow(f'focus_area_{i}', focus_area)

    cv2.imshow('f', sample_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    
        

