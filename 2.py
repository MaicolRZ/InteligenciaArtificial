import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

# Cargar modelo y pesos
modelo = 'ModeloRCNNv5.h5'
peso =  'pesosRCNNv5.h5'
cnn = load_model(modelo)
cnn.load_weights(peso)

#Clases del modelo entrenado
CLASES = {0: 'Gracias', 1: 'Adios', 2: 'Hola', 3: 'Dolor', 4: 'Equivocarse', 5: 'Te Quiero', 6: 'Yo Soy', 7: 'Necesitar', 8: 'Ayuda', 9: 'Doctor'}

# Inicializar la clase MediaPipe Hands
mp_hands = mp.solutions.hands

# Configurar la detección de manos
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)

# Inicializar la clase de dibujo de MediaPipe
mp_draw = mp.solutions.drawing_utils

# Inicializar la cámara
cap = cv2.VideoCapture(0)
def get_hand_box(hand_landmarks, img, box_size=250):
    """Obtener las coordenadas (x, y) del cuadro que rodea la mano"""
    x_min, y_min, x_max, y_max = img.shape[1], img.shape[0], 0, 0
    
    # Obtener las coordenadas de los puntos de la mano
    landmark_list = []
    for i, landmark in enumerate(hand_landmarks.landmark):
        h, w, c = img.shape
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        landmark_list.append([cx, cy])
        
        # Actualizar las coordenadas del cuadro
        if cx < x_min:
            x_min = cx
        if cy < y_min:
            y_min = cy
        if cx > x_max:
            x_max = cx
        if cy > y_max:
            y_max = cy
    
    # Obtener el centro del cuadro
    center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
    
    # Obtener las coordenadas del cuadro con el tamaño deseado
    x_min = max(0, center_x - box_size // 2)
    y_min = max(0, center_y - box_size // 2)
    x_max = min(img.shape[1], center_x + box_size // 2)
    y_max = min(img.shape[0], center_y + box_size // 2)
    
    return x_min, y_min, x_max, y_max
contador = 0
suma = 0
while True:
    # Leer la imagen de la cámara
    success, img = cap.read()
    if not success:
        print("No se pudo obtener el video de la cámara.")
        break
    # Convertir la imagen a BGR a RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detectar manos en la imagen
    results = hands.process(img_rgb)

    # Comprobar si se han detectado manos
    if results.multi_hand_landmarks:
        # Dibujar los puntos de la mano en la imagen
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                     landmark_drawing_spec=mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                     connection_drawing_spec=mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2,circle_radius=2),
                     )

        # Dibujar un cuadro alrededor de la mano o manos
        num_hands = len(results.multi_hand_landmarks)
        if num_hands == 1:
            hand_landmarks = results.multi_hand_landmarks[0]
            x_min, y_min, x_max, y_max = get_hand_box(hand_landmarks, img, box_size=250)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 255, 255), 3)
        elif num_hands == 2:
            hand_landmarks1 = results.multi_hand_landmarks[0]
            hand_landmarks2 = results.multi_hand_landmarks[1]
            x_min1, y_min1, x_max1, y_max1 = get_hand_box(hand_landmarks1, img)
            x_min2, y_min2, x_max2, y_max2 = get_hand_box(hand_landmarks2, img)
            x_min, y_min = min(x_min1, x_min2), min(y_min1, y_min2)
            x_max, y_max = max(x_max1, x_max2), max(y_max1, y_max2)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 255, 255), 3)
        
        roi = img[y_min+3:y_max-3, x_min+3:x_max-3]
        if roi.size != 0:
            
            # Preprocesar la ROI de la imagen
            roi = cv2.resize(roi, (300, 300))
            roi = roi.astype("float") / 255.0
            roi_gray_neg = 1.0 - roi
            cv2.imshow("ROI", roi_gray_neg)
            roi = np.expand_dims(roi, axis=0)
            predicciones = cnn.predict(roi,verbose=0)
            clase = CLASES[np.argmax(predicciones)]


            contador += 1
            suma += predicciones[0][np.argmax(predicciones)]

            if contador == 10:
                promedio = suma / 10
                print("Promedio:", promedio)

                contador = 0
                suma = 0

            prob_max = np.max(predicciones)
            porcentaje_val = float(prob_max*100)
            porcentaje=round(porcentaje_val, 3)
            if porcentaje < 80:
                    color =(255,255,255)
            elif porcentaje >= 80:
                color =(0,255,0) 
                text = f"{clase}: {predicciones[0][np.argmax(predicciones)]*100:.2f}%"
                cv2.rectangle(img,(x_min, y_min-40), (x_max, y_min), (255, 255, 255), -1)
                cv2.putText(img, text, (x_min + 10, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    # Mostrar la imagen en una ventana
            cv2.imshow("Imagen", img) 
    else:
        # mostrar la imagen original sin cuadrado ni puntos de referencia
        cv2.imshow("Imagen", img)  
    # Esperar a que se pulse la tecla 'q' para salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar la ventana
cap.release()
cv2.destroyAllWindows()


