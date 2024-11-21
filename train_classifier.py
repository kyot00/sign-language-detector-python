import os
import cv2
import mediapipe as mp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

DATA_DIR = './data'
data = []
labels = []

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

expected_landmarks = 42  # 21 puntos de referencia por mano, 2 coordenadas (x, y) por punto

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        img_full_path = os.path.join(DATA_DIR, dir_, img_path)
        img = cv2.imread(img_full_path)
        if img is None:
            print(f"Error al leer la imagen {img_path} en la clase {dir_}")
            os.remove(img_full_path)
            continue

        # Preprocesar la imagen
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (640, 480))  # Redimensionar la imagen

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)

            # Verificar si data_aux tiene la longitud esperada
            if len(data_aux) == expected_landmarks:
                data.append(data_aux)
                labels.append(dir_)
            else:
                print(f"Longitud inesperada de data_aux en {img_path} de la clase {dir_}")
                os.remove(img_full_path)
        else:
            print(f"No se detectaron manos en {img_path} de la clase {dir_}")
            os.remove(img_full_path)

# Verificar si hay datos antes de continuar
if not data or not labels:
    print("No se encontraron datos suficientes para entrenar el modelo.")
    exit()

# Verificar las clases únicas en labels
unique_labels = set(labels)
print(f"Clases únicas encontradas durante el entrenamiento: {unique_labels}")

# Convertir labels a valores numéricos
label_map = {label: idx for idx, label in enumerate(unique_labels)}
numeric_labels = [label_map[label] for label in labels]

# Normalizar los datos
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(data, numeric_labels, test_size=0.2, random_state=84)

# Entrenar el modelo
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Evaluar el modelo
accuracy = model.score(X_test, y_test)
print(f"Precisión del modelo: {accuracy * 100:.2f}%")

# Guardar el modelo, el escalador y el label_map
with open('model.p', 'wb') as f:
    pickle.dump({'model': model, 'scaler': scaler, 'label_map': label_map}, f)
