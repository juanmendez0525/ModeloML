import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# === Rutas de las imágenes ===
train_dir = "dataset/train"

# === Generador de imágenes con aumento de datos ===
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 80% entrenamiento, 20% validación
)

# === Cargar datos ===
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary',
    subset='validation'
)

# === Crear modelo CNN ===
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# === Compilar modelo ===
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# === Entrenar modelo ===
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=5
)

# === Guardar modelo entrenado ===
model.save("modelo_perros_gatos.h5")
print("✅ Modelo guardado como modelo_perros_gatos.h5")

# === Graficar resultados ===
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión del modelo Perro vs Gato')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.show()
