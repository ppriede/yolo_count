# 👀📒👓 yolo_count
Se simplifican varias cosas con ultima version de ultralytics yolov8 
Roboflow saco video mostrando esto, hace conteo y ocupa los modelo de ultralytics para tracking

https://www.youtube.com/watch?v=Mi9iHFd0_Bo
https://github.com/SkalskiP/yolov8-native-tracking

## Instalar
```sh
pip install ultralytics
pip install supervision
```

### Versiones
```sh
yolo version
# 8.0.57
pip show supervision
# Name: supervision
# Version: 0.3.2
# Summary: A set of easy-to-use utils that will come in handy in any Computer Vision project
# Home-page: https://github.com/roboflow/supervision
# Author: Piotr Skalski
# Author-email: piotr.skalski92@gmail.com
# License: MIT
# Location: /home/stan/DEV/venv_yolo/lib/python3.8/site-packages
# Requires: opencv-python, numpy, matplotlib
# Required-by:

```

## 💻 Prueba rapida
Para ejecutar en CLI y ver que todo funciona correctamente

Lista de elementos configurable 
https://docs.ultralytics.com/usage/cfg/#prediction

```sh
yolo track model="yolov8n.pt" source="../seg1_corto.mp4"
yolo track model="yolov8n.pt" source="../seg1_corto.mp4" save=True save_txt=True save_conf=True save_crop=True
yolo track model="yolov8n.pt" source="../seg1_corto.mp4" conf=0.7 save=True save_txt=True save_conf=True save_crop=True
yolo track model="yolov8n.pt" source="../seg1_corto.mp4" conf=0.7 save=True save_txt=True save_conf=True save_crop=True classes=[2,3,5,6,7]
yolo track model="yolov8x-seg.pt" source="../seg1_corto.mp4" conf=0.7 save=True save_txt=True save_conf=True save_crop=True classes=[2,3,5,6,7]
yolo track model="yolov8x.pt" source="../seg1_corto.mp4" conf=0.7 save=True save_txt=True save_conf=True save_crop=True classes=[2,3,5,6,7]
yolo track model="yolov8x.pt" source="../seg1_corto.mp4" conf=0.5 save=True save_txt=True save_conf=True save_crop=True classes=[2,3,5,6,7]
```
Todo en orden, continuamos con Python

# Python (SDK)

## main.py 🐍
Modificamos las lineas necesarias

Para el video usado, queremos que la linea pase por:
Punto 1 (122, 518)
Punto 2: (516, 487)

No esta pescando bien la primera parte, seguimos despues.

