import os
from ultralytics import YOLO
import supervision as sv
import numpy as np
# Carpeta de trabajo
HOME = os.getcwd()
# Video para analizar
SOURCE_VIDEO_PATH = f"{HOME}/../seg1_corto.mp4"
# Video resultado final
TARGET_VIDEO_PATH = f"{HOME}/seg1_yolo_count.mp4"
# Linea de deteccion
LINE_START = sv.Point(122, 518)
LINE_END = sv.Point(516, 487)
# Informacion de video original
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
print(video_info)
duracion = video_info.total_frames/video_info.fps
print("Duracion:",duracion,"segundos")
#exit()
def main():
    line_counter = sv.LineZone(
        start=LINE_START,
        end=LINE_END)
    line_annotator = sv.LineZoneAnnotator(
        thickness=5,
        text_thickness=1,
        text_scale=1)
    box_annotator = sv.BoxAnnotator(
        thickness=1,
        text_thickness=1,
        text_scale=1
    )
    model = YOLO("yolov8n.pt")
    i=0
    with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
        for result in model.track(
            source=SOURCE_VIDEO_PATH,
            save=False,
            save_txt=False,
            save_conf=False,
            save_crop=False,
            hide_labels=True,
            hide_conf=True,
            visualize=False,
            boxes=False,
            show=False,
            conf=0.25,
            device=0,
            classes=[2,3,5,6,7],
            verbose=True):
            i += 1
            frame = result.orig_img
            detections = sv.Detections.from_yolov8(result)
            if result.boxes.id is not None:
                detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
            labels = [
                f"{tracker_id} {model.model.names[class_id]} {class_id} {confidence:0.2f}"
                for _, confidence, class_id, tracker_id
                in detections
            ]
            frame = box_annotator.annotate(
                scene=frame, 
                detections=detections,
                labels=labels)
            line_counter.trigger(detections=detections)
            print("IN: ",line_counter.in_count, "i: ",i)
            line_annotator.annotate(frame=frame, line_counter=line_counter)
            sink.write_frame(frame=frame)


if __name__ == "__main__":
    main()