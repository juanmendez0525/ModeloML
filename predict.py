from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# === Cargar modelo ===
model = load_model("modelo_perros_gatos.h5")

# === Ruta de la imagen de prueba ===
ruta_imagen = "imagen-de-prueba/prueba.jpg"

# === Preparar imagen ===
img = image.load_img(ruta_imagen, target_size=(150, 150))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# === Realizar predicción ===
prediccion = model.predict(img_array)[0][0]

# === Mostrar resultado ===
if prediccion > 0.5:
    etiqueta = "🐶 Es un PERRO"
else:
    etiqueta = "🐱 Es un GATO"

print(f"Predicción: {etiqueta}")

# === Mostrar imagen con el resultado ===
plt.imshow(image.load_img(ruta_imagen))
plt.title(etiqueta)
plt.axis("off")
plt.show()
