import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from ultralytics import YOLO
from roboflow import Roboflow
import os

from roboflow import Roboflow
rf = Roboflow(api_key="gKYxCHMXIdbNI94Ohan8")
project = rf.workspace("boladepapel").project("deteccao-lixo-vxqjl")
version = project.version(1)
dataset = version.download("yolov8")
                

# Carrega um modelo YOLOv8 pré-treinado (yolov8n.pt é o menor e mais rápido)
model = YOLO('yolov8n.pt')

# Treina o modelo com o seu dataset
if __name__ == '__main__':
    results = model.train(
        data=os.path.join(dataset.location, 'data.yaml'),  # Caminho para o arquivo de configuração do dataset
        epochs=50,  # Número de épocas de treinamento. 50 é um bom começo.
        imgsz=640   # Tamanho da imagem para o treinamento
    )