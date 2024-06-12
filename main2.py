import cv2
from roboflow import Roboflow
import time

# Inicializar Roboflow
rf = Roboflow(api_key="0A38R8UDuFt11vAHykfS")

# Cargar proyecto y versión
project = rf.workspace("test-pgigc").project("carrots-j885y")
version = project.version(1)

# Inicializar modelo
model = version.model

# Capturar video desde la cámara web
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

# Configurar la resolución deseada (por ejemplo, 640x480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    # Capturar cada fotograma
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo capturar el fotograma.")
        break

    # Guardar el fotograma en un archivo temporal
    cv2.imwrite('temp_frame.jpg', frame)

    # Inferir en el fotograma guardado
    start_time = time.time()
    prediction = model.predict('temp_frame.jpg', confidence=20, overlap=40).json()
    print(f"Tiempo de inferencia: {time.time() - start_time:.2f} segundos")

    # Verificar si la predicción tiene el campo 'predictions'
    if 'predictions' in prediction:
        # Dibujar las predicciones en el fotograma
        for pred in prediction['predictions']:
            x0, y0 = int(pred['x'] - pred['width'] / 2), int(pred['y'] - pred['height'] / 2)
            x1, y1 = int(pred['x'] + pred['width'] / 2), int(pred['y'] + pred['height'] / 2)
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(frame, pred['class'], (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mostrar el fotograma con las predicciones
    cv2.imshow('Detección en tiempo real', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar el objeto de captura y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
