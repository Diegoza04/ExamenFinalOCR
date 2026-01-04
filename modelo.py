import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def construir_modelo():
    """
    Construye un modelo simple de CNN para reconocimiento de caracteres.
    :return: Modelo construido.
    """
    modelo = Sequential([
        # Primera capa convolucional
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Segunda capa convolucional
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten y capas densas para clasificación
        Flatten(),
        Dense(128, activation='relu'),
        Dense(64, activation='softmax')  # 10 números + 26 letras mayúsculas + Ñ
    ])
    modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return modelo

def entrenar_modelo(modelo, x_train, y_train, x_val, y_val, epochs=10):
    """
    Entrena el modelo con los datos proporcionados.
    :param modelo: El modelo CNN que será entrenado.
    :param x_train: Imágenes de entrenamiento.
    :param y_train: Etiquetas de entrenamiento.
    :param x_val: Imágenes de validación.
    :param y_val: Etiquetas de validación.
    :param epochs: Número de épocas.
    :return: Historia del entrenamiento.
    """
    return modelo.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=32)

def guardar_modelo(modelo, ruta):
    """
    Guarda el modelo en la ruta especificada.
    :param modelo: Modelo a guardar.
    :param ruta: Ruta donde guardar el modelo.
    """
    modelo.save(ruta)
    print(f"Modelo guardado en {ruta}.")

def cargar_modelo(ruta):
    """
    Carga un modelo desde la ruta especificada.
    :param ruta: Ruta del modelo.
    :return: Modelo cargado.
    """
    return tf.keras.models.load_model(ruta)