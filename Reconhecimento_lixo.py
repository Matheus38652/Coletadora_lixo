import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import os
import json
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from PIL import Image, ImageTk

model = None
class_indices = None
cap = None

def verificar_estrutura_pasta(pasta_imagens):
    """Verifica se a pasta de imagens tem subpastas com arquivos, como esperado pelo Keras."""
    if not os.path.exists(pasta_imagens): return False
    subpastas = [f.path for f in os.scandir(pasta_imagens) if f.is_dir()]
    if len(subpastas) == 0: return False
    for subpasta in subpastas:
        if len([f for f in os.scandir(subpasta) if f.is_file()]) == 0: return False
    return True

def criar_modelo_com_transfer_learning(numero_de_classes):
    """Cria um modelo de classificação de imagem usando MobileNetV2 como base."""
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(numero_de_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def treinar_modelo(pasta_imagens):
    """Função para treinar o modelo de reconhecimento de lixo."""
    global model, class_indices
    if not verificar_estrutura_pasta(pasta_imagens):
        messagebox.showerror("Erro de Estrutura", "A pasta selecionada deve conter subpastas (uma para cada tipo de lixo) com imagens dentro.")
        return
    
    print("Iniciando o treinamento do modelo de reconhecimento de lixo...")
    messagebox.showinfo("Treinamento", "O treinamento foi iniciado. Por favor, aguarde a mensagem de conclusão. Isso pode levar vários minutos.")

    # Aumenta a variedade das imagens de treino para o modelo generalizar melhor
    datagen = ImageDataGenerator(
        rescale=1./255, 
        rotation_range=40, 
        width_shift_range=0.2, 
        height_shift_range=0.2, 
        shear_range=0.2, 
        zoom_range=0.2, 
        horizontal_flip=True, 
        fill_mode='nearest'
    )
    train_generator = datagen.flow_from_directory(
        pasta_imagens, 
        target_size=(224, 224), 
        batch_size=32, 
        class_mode='categorical'
    )

    model = criar_modelo_com_transfer_learning(len(train_generator.class_indices))
    
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_generator, epochs=20) # 20 épocas é um bom começo
    
    # Salva o modelo treinado e os nomes das classes
    model.save('modelo_lixo.h5')
    with open('classes_lixo.json', 'w') as f:
        json.dump(train_generator.class_indices, f)

    acc_final = history.history['accuracy'][-1] * 100
    messagebox.showinfo("Treinamento Concluído", f"Modelo treinado com precisão final de {acc_final:.2f}% e salvo com sucesso!")

def carregar_modelo_existente():
    """Carrega o modelo de lixo previamente treinado."""
    global model, class_indices
    try:
        model = load_model('modelo_lixo.h5')
        with open('classes_lixo.json', 'r') as f:
            class_indices = json.load(f)
        return True
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return False

def reconhecer_lixo(frame, class_indices):
    """Processa um frame da câmera e retorna a previsão do modelo."""
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image).resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    
    previsao = model.predict(image)
    classe_index = np.argmax(previsao)
    confianca = np.max(previsao) * 100
    
    indices_para_classes = {v: k for k, v in class_indices.items()}
    nome_classe = indices_para_classes[classe_index]
    
    # Classifica tudo que for reconhecido como "Lixo" por enquanto
    if confianca > 60: # Limiar de confiança para exibir a previsão
        return f"Lixo ({nome_classe})", confianca
    return "Nenhum lixo detectado", 0

def atualizar_imagem(label_img):
    """Loop principal que captura frames da câmera e exibe as previsões."""
    global cap
    if cap and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if model and class_indices:
                previsao_texto, confianca = reconhecer_lixo(frame, class_indices)
                
                # Formata o texto para exibição no vídeo
                texto_final = f'{previsao_texto.capitalize()} [{confianca:.1f}%]'
                cor = (0, 255, 0) if "Lixo" in previsao_texto else (0, 0, 255)

                cv2.rectangle(frame, (5, 5), (550, 40), (0, 0, 0), -1)
                cv2.putText(frame, texto_final, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2)

            # Converte e exibe a imagem na interface Tkinter
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(img_pil)
            label_img.config(image=img_tk)
            label_img.image = img_tk

        label_img.after(10, lambda: atualizar_imagem(label_img))

def ativar_camera(label_img):
    global cap
    if carregar_modelo_existente():
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                messagebox.showerror("Erro de Câmera", "Não foi possível acessar a câmera.")
                cap = None
                return
            atualizar_imagem(label_img)
    else:
        messagebox.showwarning("Modelo não encontrado", "Você precisa treinar o modelo primeiro antes de iniciar a câmera.")


def selecionar_pasta_e_treinar():
    pasta_imagens = filedialog.askdirectory()
    if pasta_imagens:
        treinar_modelo(pasta_imagens)

def interface_grafica():
    """Cria a janela principal da aplicação."""
    root = tk.Tk()
    root.title("Coletor de Lixo - IA de Reconhecimento")
    root.geometry("700x600")

    btn_treinar = tk.Button(root, text="Treinar Modelo", command=selecionar_pasta_e_treinar)
    btn_treinar.pack(pady=10)
    
    label_img = tk.Label(root)
    label_img.pack(pady=10)

    btn_camera = tk.Button(root, text="Iniciar Câmera e Coleta", command=lambda: ativar_camera(label_img))
    btn_camera.pack(pady=5)
    

    root.mainloop()

    if cap:
        cap.release()

if __name__ == "__main__":
    interface_grafica()