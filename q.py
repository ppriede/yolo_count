import os
import argparse
from ultralytics import YOLO
import supervision as sv
import numpy as np


# Carpeta de trabajo
HOME = os.getcwd()
# Despues seguimos mirando las instrucciones de argparse
# https://machinelearningmastery.com/command-line-arguments-for-your-python-script/
# parser = argparse.ArgumentParser(description="Proceso de video con Ultralytics YOLOv8 + Supervision",
#                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument("entrada", help="Entrade de Imagen (JPG, PNG, MP4, MKV, etc)", default=f"{HOME}/../seg1_corto.mp4")
# args = parser.parse_args()
# config = vars(args)
# print(config)
# print(config.entrada)
# exit()

# Video para analizar
SOURCE_VIDEO_PATH = f"{HOME}/../PMC01_09_a_11.mkv"
# TARGET_VIDEO_PATH = cfg.source if cfg.source is not None else TARGET_VIDEO_PATH
# Video resultado final
TARGET_VIDEO_PATH = f"{HOME}/seg1_corto_stream_T_yolo_count.mp4"
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
    i=0 # Cuadro inicial
    # Si calculamos el tiempo, obtenemos el tiempo del cuadro
    # i/fps
    file_object = open(f"{HOME}/conteo_largo.txt", 'a')
    for result in model.track(
        source=SOURCE_VIDEO_PATH, # Ruta de archivo a leer, si es 0 es webcam o camara conectada
        save=False, # Guarda video
        save_txt=True, # Guarda la informacion del cuadro donde se detecto algo
        save_conf=True, # Guarda el intervalo de confianza junto con los datos
        save_crop=False, # Guarda una imagen de lo detectado
        hide_labels=True, # Oculta la etiqueta de la deteccion
        hide_conf=True, # Oculta el intervalo de confianza en la imagen
        visualize=False, # No se
        boxes=False, # Muestra las cajas alrededor del objeto
        show=False, # Si puede, muestra informacion
        conf=0.25, # Intervalo de confianza para la deteccion
        device=0, # Usamos device=0 (GPU)
        classes=[2,3,5,6,7], # Solo identificamos estas clases (coco.txt con lista completa)
        verbose=False, #Esto genera mas informacion en CLI
        stream=True # Esto, al parecer, entrega los datos en cada deteccion... veamos si lo entiendo bien
        ):
        tiempo =i/video_info.fps
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
        print("IN: ",line_counter.in_count, "i: ",i,"Tiempo: ",tiempo,"segundos")
        
        # file_object.write(str(line_counter.in_count))
        # file_object.write(";")
        # file_object.write(str(i))
        # file_object.write(";")
        # file_object.write(str(tiempo))
        # file_object.write(";")
        # file_object.write(str(labels))
        # file_object.write(";")
        # file_object.write(str(detections))
        # file_object.write("\n")
        line_annotator.annotate(frame=frame, line_counter=line_counter)
        i+=1

if __name__ == "__main__":
    main()