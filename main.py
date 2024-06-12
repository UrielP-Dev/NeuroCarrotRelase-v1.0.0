from roboflow import Roboflow

# Inicializar Roboflow
rf = Roboflow(api_key="0A38R8UDuFt11vAHykfS")

# Cargar proyecto y versión
project = rf.workspace("test-pgigc").project("carrots-j885y")
version = project.version(1)
dataset = version.download("yolov8")

# Inicializar modelo
model = version.model

# Inferir en una imagen local y guardar la predicción visualizada
prediction = model.predict("2024-04-Int.-Carrot-Day.webp", confidence=60, overlap=40)
prediction.save("prediction.jpg")

# Imprimir la predicción junto con el nombre del archivo
print(f"Predicción para '2024-04-Int.-Carrot-Day.webp': {prediction}")

# Inferir en una imagen alojada en otro lugar (opcional)
# hosted_prediction = model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30)
# hosted_prediction.save("hosted_prediction.jpg")

print("Predicción guardada en prediction.jpg")
