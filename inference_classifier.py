import pickle
import cv2
import mediapipe as mp
import numpy as np

# Cargar el modelo entrenado y el escalador desde 'model.p'
with open('./model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']
scaler = model_dict['scaler']
label_map = model_dict['label_map']

# Invertir el label_map para obtener el diccionario de etiquetas
labels_dict = {v: k for k, v in label_map.items()}

cap = cv2.VideoCapture(1)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=2)

expected_landmarks = 42  # Asegúrate de que este valor sea consistente con el entrenamiento

while cap.isOpened():
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        print("Error al capturar la imagen de la cámara.")
        break

    H, W, _ = frame.shape

    # Procesar la imagen
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)
                data_aux.append(x)
                data_aux.append(y)

        # Calcular las coordenadas del rectángulo
        x1 = int(min(x_) * W)
        y1 = int(min(y_) * H)
        x2 = int(max(x_) * W)
        y2 = int(max(y_) * H)

        # Verificar si data_aux tiene la longitud esperada
        if len(data_aux) == expected_landmarks:
            # Normalizar los datos
            data_aux = np.array(data_aux).reshape(1, -1)
            data_aux = scaler.transform(data_aux)

            # Predecir la clase y obtener la probabilidad
            prediction = model.predict(data_aux)
            predicted_label = labels_dict[prediction[0]]
            probabilities = model.predict_proba(data_aux)
            confidence = np.max(probabilities) * 100  # Obtener la probabilidad más alta

            print(f"Predicción: {predicted_label} con {confidence:.2f}% de confianza")

            # Dibujar el rectángulo y la predicción en el frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'{predicted_label} ({confidence:.2f}%)', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        else:
            print(f"Longitud inesperada de data_aux: {len(data_aux)}")

        # Dibujar las manos
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()