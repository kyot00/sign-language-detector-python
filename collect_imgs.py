import os
import cv2
import mediapipe as mp

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Definir las etiquetas de las clases en el orden específico y en mayúsculas
class_labels = [
'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'ENIE', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]
dataset_size = 300

cap = cv2.VideoCapture(1)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)


for label in class_labels:
    class_dir = os.path.join(DATA_DIR, label)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(label))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar la imagen de la cámara.")
            break

        # Ajustar brillo y contraste
        alpha = 1.2  # Contraste
        beta = -50    # Brillo
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
 
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # Crear una copia del frame para la previsualización
        preview_frame = frame.copy()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_ = [landmark.x for landmark in hand_landmarks.landmark]
                y_ = [landmark.y for landmark in hand_landmarks.landmark]

                x1 = int(min(x_) * frame.shape[1]) - 20  # Añadir margen
                y1 = int(min(y_) * frame.shape[0]) - 20  # Añadir margen
                x2 = int(max(x_) * frame.shape[1]) + 20  # Añadir margen
                y2 = int(max(y_) * frame.shape[0]) + 20  # Añadir margen

                cv2.rectangle(preview_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                mp_drawing.draw_landmarks(
                    preview_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        cv2.putText(preview_frame, 'Ready? Press "Q" to start, "Z" to cancel', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', preview_frame)
        key = cv2.waitKey(25)
        if key == ord('Q'):
            break
        elif key == ord('Z'):
            print("Cancelando la toma de muestras.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar la imagen de la cámara.")
            break

        # Ajustar brillo y contraste
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # Crear una copia del frame para la previsualización
        preview_frame = frame.copy()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_ = [landmark.x for landmark in hand_landmarks.landmark]
                y_ = [landmark.y for landmark in hand_landmarks.landmark]

                x1 = int(min(x_) * frame.shape[1]) - 20  # Añadir margen
                y1 = int(min(y_) * frame.shape[0]) - 20  # Añadir margen
                x2 = int(max(x_) * frame.shape[1]) + 20  # Añadir margen
                y2 = int(max(y_) * frame.shape[0]) + 20  # Añadir margen

                cv2.rectangle(preview_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                mp_drawing.draw_landmarks(
                    preview_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        cv2.imshow('frame', preview_frame)
        key = cv2.waitKey(25)
        if key == ord('Z'):
            print("Cancelando la toma de muestras.")
            cap.release()
            cv2.destroyAllWindows()
            exit()
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()