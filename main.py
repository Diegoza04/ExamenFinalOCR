import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from modelo import construir_modelo, guardar_modelo, cargar_modelo
from dataset import cargar_dataset
from preprocesamiento import procesar_imagen


def predecir_caracter(ruta_imagen, modelo, label_map):
    """
    Realiza una predicción sobre la imagen dada y devuelve el carácter correspondiente.
    :param ruta_imagen: Ruta a la imagen de entrada.
    :param modelo: Modelo entrenado.
    :param label_map: Mapa de etiquetas para decodificar la predicción.
    :return: Predicción del carácter.
    """
    try:
        # Procesar la imagen
        imagen_procesada = procesar_imagen(ruta_imagen)

        if imagen_procesada is None:
            return "Error: No se pudo procesar la imagen."

        # Expandir dimensiones para formato de lote
        imagen_procesada = np.expand_dims(imagen_procesada, axis=0)

        # Obtener predicciones del modelo
        predicciones = modelo.predict(imagen_procesada)
        indice = np.argmax(predicciones)

        # Obtener el carácter correspondiente
        caracter = label_map.get(indice, "Desconocido")
        return caracter
    except Exception as e:
        return f"Error durante la predicción: {e}"


def entrenar_modelo_mejorado(x_train, y_train, x_val, y_val, numero_clases, guardado_path):
    """
    Entrena un modelo mejorado con data augmentation y class weights.
    :param x_train: Datos de entrenamiento.
    :param y_train: Etiquetas de entrenamiento.
    :param x_val: Datos de validación.
    :param y_val: Etiquetas de validación.
    :param numero_clases: Número total de clases.
    :param guardado_path: Ruta para guardar el modelo entrenado.
    :return: Modelo entrenado.
    """
    # Balancear las clases con class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,       # Rotaciones pequeñas
        width_shift_range=0.1,   # Desplazamientos horizontales
        height_shift_range=0.1,  # Desplazamientos verticales
        zoom_range=0.2           # Zoom
    )
    datagen.fit(x_train)

    # Construir modelo mejorado
    modelo = construir_modelo()
    modelo.fit(
        datagen.flow(x_train, y_train, batch_size=32),
        validation_data=(x_val, y_val),
        epochs=30,
        class_weight=class_weights
    )

    # Guardar el modelo entrenado
    guardar_modelo(modelo, guardado_path)
    print(f"Modelo mejorado guardado en: {guardado_path}")

    return modelo


def predecir_caracteres_en_carpeta(ruta_carpeta, modelo, ruta_archivo_resultados, label_map):
    """
    Recorre una carpeta, reconoce cada imagen y guarda los resultados en un archivo.
    :param ruta_carpeta: Ruta de la carpeta con imágenes para procesar.
    :param modelo: Modelo entrenado.
    :param ruta_archivo_resultados: Ruta del archivo donde se guardarán los resultados.
    :param label_map: Mapa que convierte índices en caracteres.
    """
    # Verificar si la carpeta existe
    if not os.path.isdir(ruta_carpeta):
        print(f"Error: La carpeta '{ruta_carpeta}' no se encontró.")
        return

    print(f"Procesando todas las imágenes en la carpeta: {ruta_carpeta}")

    # Abrir el archivo de resultados
    with open(ruta_archivo_resultados, "w", encoding="utf-8") as f:
        f.write("Resultados de las predicciones:\n")
        f.write("--------------------------------\n")

        # Recorrer todos los archivos en la carpeta
        for nombre_archivo in sorted(os.listdir(ruta_carpeta)):
            ruta_imagen = os.path.join(ruta_carpeta, nombre_archivo)

            # Verificar que el archivo sea una imagen
            if not nombre_archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                print(f"Archivo ignorado (no es imagen): {nombre_archivo}")
                f.write(f"{nombre_archivo}: No se procesó (archivo no es imagen válida).\n")
                continue

            # Realizar la predicción para la imagen
            resultado = predecir_caracter(ruta_imagen, modelo, label_map)
            print(f"Imagen: {nombre_archivo} -> Carácter predicho: {resultado}")

            # Guardar el resultado en el archivo
            f.write(f"{nombre_archivo}: {resultado}\n")

    print(f"\nResultados guardados en: {ruta_archivo_resultados}")


if __name__ == "__main__":
    # Configura si vas a entrenar un modelo nuevo o cargar uno existente
    entrenamiento = False

    # Etiquetas del dataset (LABEL_MAP)
    LABEL_MAP = {
        **{i: str(i) for i in range(10)},                              # Números: 0-9
        **{i + 10: chr(i + ord('A')) for i in range(26)},             # Mayúsculas: A-Z
        36: "Ñ",
        **{i + 37: chr(i + ord('a')) for i in range(26)},             # Minúsculas: a-z
        63: "ñ"
    }

    if entrenamiento:
        print("Cargando y procesando el dataset...")

        # Cargar dataset
        x, y = cargar_dataset("DatasetsOCR25-26")

        # Dividir en entrenamiento y validación
        split = int(0.8 * len(x))
        x_train, y_train = x[:split], y[:split]
        x_val, y_val = x[split:], y[split:]

        # Crear y entrenar el modelo mejorado
        modelo = entrenar_modelo_mejorado(x_train, y_train, x_val, y_val, len(LABEL_MAP), "modelo_mejorado_ocr.h5")

    else:
        print("Cargando modelo entrenado...")
        modelo = cargar_modelo("modelo_mejorado_ocr.h5")
        print("Modelo cargado con éxito.")

        # Ruta a la carpeta con imágenes a procesar
        ruta_carpeta_imagenes = r"C:\Users\barra\.vscode\Inteligencia Artificial\ExamenFinal\ExamenFinalOCR\DatasetsOCR25-26\mayusculas\A"

        # Ruta al archivo donde se guardarán los resultados
        ruta_archivo_resultados = r"C:\Users\barra\.vscode\Inteligencia Artificial\ExamenFinal\ExamenFinalOCR\resultados.txt"

        # Procesar todas las imágenes en la carpeta
        predecir_caracteres_en_carpeta(ruta_carpeta_imagenes, modelo, ruta_archivo_resultados, LABEL_MAP)