# detector_lixo_yolo.py

import cv2
from ultralytics import YOLO

# Carregue o modelo YOLOv8 que você treinou
# ATENÇÃO: Atualize este caminho para onde o seu modelo 'best.pt' foi salvo!
model_path = 'runs/detect/train/weights/best.pt'
model = YOLO(model_path)

# Inicia a captura de vídeo da câmera padrão
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro: Não foi possível abrir a câmera.")
    exit()

while True:
    # Captura um frame da câmera
    success, frame = cap.read()

    if success:
        # Executa a detecção do YOLOv8 no frame
        results = model(frame)

        # Plota os resultados no frame (desenha os boxes, classes e confianças)
        annotated_frame = results[0].plot()

        # Exibe o frame anotado em uma janela
        cv2.imshow("Detector de Lixo YOLOv8", annotated_frame)

        # Se a tecla 'q' for pressionada, interrompe o loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Se não conseguir capturar o frame, interrompe o loop
        break

# Libera os recursos da câmera e fecha todas as janelas
cap.release()
cv2.destroyAllWindows()