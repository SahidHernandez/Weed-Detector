# Detección de Maleza con Visión Artificial para Aplicaciones Agrícolas 🌿🤖

Este proyecto presenta el desarrollo de un sistema de detección automática de maleza en imágenes agrícolas, utilizando algoritmos de visión por computadora basados en YOLOv11 y TensorFlow (SSD MobileNet V2). La aplicación también incluye una interfaz gráfica amigable que permite realizar detección tanto en imágenes estáticas como en tiempo real con cámara web.

## 📌 Objetivo

Desarrollar un sistema de detección de maleza en campos agrícolas mediante el uso de redes neuronales convolucionales para reducir el uso de pesticidas, optimizar recursos y facilitar el monitoreo en tiempo real.

---

## 📦 Tecnologías Utilizadas

- **Lenguaje:** Python 3.9
- **Modelos:**
  - YOLOv11 (`ultralytics`)
  - TensorFlow Object Detection API (SSD MobileNet V2)
- **Frameworks / Librerías:**
  - TensorFlow 2.10.0
  - OpenCV
  - Tkinter (interfaz gráfica)
  - NumPy, PIL, Protobuf
- **Aceleración GPU:** CUDA 11.8, cuDNN 8.6

---

## 📁 Dataset

- **Nombre:** [Weeds Dataset](https://universe.roboflow.com/augmented-startups/weeds-nxe1w)
- **Fuente:** Roboflow
- **Tamaño:** 4203 imágenes
- **Formato:** YOLO11 y TFRecord

---

## 🛠️ Instalación y Configuración

Para utilizar el modelo de YOLO, necesitaras instalar la libreria Ultralytics:

`pip install ultralytics`

Para utilizar el modelo de Tensorflow, seguir las intrucciones de `guia_entrenamiento_tensorflow.txt`

# Video de demostracion

https://youtu.be/hXcxUMWstck?si=0KuyE0DyATbeGPk3
