import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from ultralytics import YOLO
from roboflow import Roboflow
import os

from roboflow import Roboflow
rf = Roboflow(api_key="gKYxCHMXIdbNI94Ohan8")
project = rf.workspace("boladepapel").project("deteccao-lixo-vxqjl")
version = project.version(2)
dataset = version.download("yolov8")
                

model = YOLO('yolov8n.pt')


if __name__ == '__main__':
    results = model.train(
        data=os.path.join(dataset.location, 'data.yaml'),
        epochs=50,
        imgsz=640,
        name='treino_lixo',
        exist_ok=True
    )