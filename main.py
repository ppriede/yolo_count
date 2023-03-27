import cv2

from ultralytics import YOLO
import supervision as sv
import numpy as np


LINE_START = sv.Point(122, 518)
LINE_END = sv.Point(516, 487)


def main():
    line_counter = sv.LineZone(
        start=LINE_START,
        end=LINE_END)
    line_annotator = sv.LineZoneAnnotator(
        thickness=2,
        text_thickness=10,
        text_scale=0.5)
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5
    )

    model = YOLO("yolov8n.pt")
    for result in model.track(
        source="../seg1.mp4",
        save=True,
        save_txt=True,
        save_conf=True,
        conf=0.3):
        
        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)

        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        
        detections = detections[(detections.class_id != 60) & (detections.class_id != 0)]

        labels = [
            f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, tracker_id
            in detections
        ]

        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections,
            labels=labels
        )

        line_counter.trigger(detections=detections)
        line_annotator.annotate(frame=frame, line_counter=line_counter)

        # cv2.imshow("yolov8", frame)

        # if (cv2.waitKey(30) == 27):
        #     break


if __name__ == "__main__":
    main()