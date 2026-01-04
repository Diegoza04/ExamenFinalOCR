import cv2
import numpy as np

def procesar_imagen(ruta_imagen):
    """
    Procesa y normaliza una imagen para convertirla en input válido para el modelo.
    :param ruta_imagen: Ruta de la imagen a procesar.
    :return: Imagen procesada como un array numpy.
    """
    IMG_SIZE = (32, 32)  # Tamaño al que redimensionaremos la imagen
    try:
        # Leer la imagen en escala de grises
        img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
        
        # Verificar que la imagen se cargó correctamente
        if img is None:
            raise ValueError("No se pudo cargar la imagen. Verifica la ruta.")
        
        # Redimensionar al tamaño objetivo
        img = cv2.resize(img, IMG_SIZE)
        
        # Binarizar la imagen (umbral)
        _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        # Normalización (valores entre 0 y 1)
        img_normalizada = img_bin / 255.0
        
        # Expandir dimensiones para cumplir formato [batch, height, width, channels]
        return np.expand_dims(img_normalizada, axis=-1)
    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        return None
    
    