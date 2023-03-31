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

## 💻 Prueba rapida Windows 10
Al parecer hay problemas con la libreria lap en Windows 10 y Python 3.9
Mejor hacer un entorno virtual en python 3.8

```sh
D:
cd ENV
conda create -n venv_yolo_p38 python=3.8
conda activate venv_yolo_p38
pip install ultralytics==8.0.51
pip install supervision==0.3.0
yolo version
# 8.0.51
pip show supervision
# Name: supervision
# Version: 0.3.0
# Summary: A set of easy-to-use utils that will come in handy in any Computer Vision project
# Home-page: https://github.com/roboflow/supervision
# Author: Piotr Skalski
# Author-email: piotr.skalski92@gmail.com
# License: MIT
# Location: d:\ppriede\anaconda3\envs\venv_yolo_p38\lib\site-packages
# Requires: matplotlib, numpy, opencv-python
# Required-by:

#Para cerrar venv
#conda deactivate
```

Pruebas con video

```sh
yolo track model="yolov8n.pt" source="seg1_corto.mp4"
```

Hay problemas en Windows10 (Workstation), no esta funcionando



## 💻 Prueba rapida Windows 10 (intento 2)
Al parecer hay problemas con la libreria lap en Windows 10 y Python 3.9
Mejor hacer un entorno virtual en python 3.8

```sh
D:
cd ENV
conda create -n venv_yolo
conda activate venv_yolo
pip install ultralytics
pip install supervision
yolo version
# 8.0.51
pip show supervision
# Name: supervision
# Version: 0.3.0
# Summary: A set of easy-to-use utils that will come in handy in any Computer Vision project
# Home-page: https://github.com/roboflow/supervision
# Author: Piotr Skalski
# Author-email: piotr.skalski92@gmail.com
# License: MIT
# Location: d:\ppriede\anaconda3\lib\site-packages
# Requires: matplotlib, numpy, opencv-python
# Required-by:

#Para cerrar venv
#conda deactivate
```

Pruebas con video

```sh
yolo track model="yolov8n.pt" source="seg1_corto.mp4"
```

Falla igual
hay un problema con lap, no instala, ni de pip ni de github
```sh
pip install lap
pip install git+git://github.com/gatagat/lap.git
```

Aqui puede existir un parche
https://github.com/ultralytics/ultralytics/issues/1328
```python
def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []

    cost_matrix[cost_matrix > thresh] = thresh + 1e-5
    indices = linear_sum_assignment(cost_matrix)
    for row, col in zip(*indices):
        if cost_matrix[row, col] > thresh:
            unmatched_a.append(row)
            unmatched_b.append(col)
        else:
            matches.append((row, col))

    return matches, unmatched_a, unmatched_b
```


# DEV de yolo_count
Montado en Ubuntu 22.04.2 LTS en WSL desde Windows 10

Varias cosas (cuda, nvidia), ya fueron instaladas de acuerdo a guias para nvidia-docker

Por simplicidad, finalmente se ocupa directamente WSL en Windows 10

Para acceder a los archivos, la ruta es ""


# Instalacion de librerias base
````sh
virtualenv venv_yolov
source venv_yolov/bin/activate
pip install ultralytics
pip install supervision
````

## Versiones
```sh
yolo version
# 8.0.58
pip show supervision
# Name: supervision
# Version: 0.3.2
# Summary: A set of easy-to-use utils that will come in handy in any Computer Vision project
# Home-page: https://github.com/roboflow/supervision
# Author: Piotr Skalski
# Author-email: piotr.skalski92@gmail.com
# License: MIT
# Location: /home/ppriede/DEV/venv_yolov/lib/python3.10/site-packages
# Requires: matplotlib, numpy, opencv-python
# Required-by:
```

## 💻 Prueba rapida

```sh
yolo track model="yolov8n.pt" source="seg1_corto.mp4"
```

### Algunas fallas
Hay que agregar cuda al PATH en WSL
https://discuss.pytorch.org/t/libcudnn-cnn-infer-so-8-library-can-not-found/164661

````sh
# Verificamos que existe ese path
ldconfig -p | grep cuda
# Agregamos
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
````

## 💻 Prueba rapida 2
Mas compleja, para guardar

```sh
yolo track model="yolov8n.pt" source="seg1_corto.mp4" conf=0.5 save=True save_txt=True save_conf=True save_crop=True classes=[2,3,5,6,7]
```

TODO EN ORDEN!

## Prueba 2 horas
Archivo Valdivia

Guardamos Video, texto, track con ID e intervalos de confianza, fotos de captura, todas las clases

```sh
yolo track model="yolov8n.pt" source="PMC01_09_a_11.mkv" conf=0.25 save=True save_txt=True save_conf=True save_crop=True device=0
```

# Python (SDK)

## main.py 🐍
Modificamos las lineas necesarias

Para el video usado, queremos que la linea pase por:
Punto 1 (122, 518)
Punto 2: (516, 487)

No esta pescando bien la primera parte, seguimos despues.
