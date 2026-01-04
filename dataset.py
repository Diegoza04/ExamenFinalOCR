import os
import numpy as np
from PIL import Image

# Etiquetas extendidas para incluir 'Ñ' y 'ñ'
LABEL_MAP = {
    **{i: str(i) for i in range(10)},                              # Números: 0-9
    **{i + 10: chr(i + ord('A')) for i in range(26)},              # Mayúsculas: A-Z
    36: "Ñ",                                                       # Mayúscula Ñ
    **{i + 37: chr(i + ord('a')) for i in range(26)},              # Minúsculas: a-z
    63: "ñ",                                                       # Minúscula ñ
}

IMG_SIZE = (32, 32)  # Redimensionar a 32x32

def cargar_dataset(ruta_dataset):
    """
    Carga imágenes y sus etiquetas desde un dataset organizado en carpetas.
    :param ruta_dataset: Ruta principal del dataset (debe contener subcarpetas).
    :return: Arrays de imágenes y etiquetas.
    """
    imagenes = []
    etiquetas = []

    print(f"Cargando dataset desde: {ruta_dataset}")

    for categoria in sorted(os.listdir(ruta_dataset)):
        ruta_categoria = os.path.join(ruta_dataset, categoria)

        # Verificar que la carpeta principal es válida
        if not os.path.isdir(ruta_categoria):
            print(f"Saltando {categoria}: No es un directorio válido.")
            continue

        print(f"Procesando categoría: {categoria}")

        for subcategoria in sorted(os.listdir(ruta_categoria)):
            ruta_subcategoria = os.path.join(ruta_categoria, subcategoria)

            # Verificar que el subdirectorio es válido
            if not os.path.isdir(ruta_subcategoria):
                print(f"Saltando {subcategoria}: No es un subdirectorio válido.")
                continue

            try:
                # Asignar etiquetas según la categoría y subcategoría
                if categoria.lower() == "numeros":
                    etiqueta = int(subcategoria)  # Etiquetas para números (0-9)
                elif categoria.lower() == "mayusculas":
                    etiqueta = 36 if subcategoria == "Ñ" else 10 + (ord(subcategoria.upper()) - ord('A'))
                elif categoria.lower() == "minusculas":
                    etiqueta = 63 if subcategoria == "ñ" else 37 + (ord(subcategoria) - ord('a'))
                else:
                    print(f"Saltando {subcategoria}: Categoría no reconocida.")
                    continue
            except ValueError as e:
                print(f"Error al asignar etiqueta para {subcategoria}: {e}")
                continue

            # Procesar imágenes dentro del subdirectorio
            for nombre_imagen in sorted(os.listdir(ruta_subcategoria)):
                ruta_imagen = os.path.join(ruta_subcategoria, nombre_imagen)

                # Verificar que el archivo es una imagen
                if not nombre_imagen.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    print(f"Saltando {nombre_imagen}: No es un archivo de imagen.")
                    continue

                try:
                    # Leer, redimensionar y normalizar la imagen
                    img = Image.open(ruta_imagen).convert("L")
                    img = img.resize(IMG_SIZE)  # Redimensionar
                    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalizar
                    img_array = np.expand_dims(img_array, axis=-1)  # Expandir dimensiones para el modelo

                    imagenes.append(img_array)
                    etiquetas.append(etiqueta)
                except Exception as e:
                    print(f"Error al procesar la imagen {nombre_imagen}: {e}")

    # Convertir listas a arrays numpy para entrenar
    return np.array(imagenes), np.array(etiquetas)


if __name__ == "__main__":
    # Ruta a la carpeta raíz del dataset
    ruta_dataset = r"DatasetsOCR25-26"

    if not os.path.exists(ruta_dataset):
        print(f"Ruta del dataset no encontrada: {ruta_dataset}")
    else:
        x, y = cargar_dataset(ruta_dataset)
        print(f"Número total de imágenes cargadas: {len(x)}")
        print(f"Número total de etiquetas cargadas: {len(y)}")