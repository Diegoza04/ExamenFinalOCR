import sys
sys.stdout.reconfigure(encoding='utf-8')

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


def load_dataset(dataset_path):
    images = []
    labels = []

    print(f"Cargando dataset desde: {dataset_path}")

    # Recorrer carpetas principales (numeros, mayusculas, minusculas)
    for category in sorted(os.listdir(dataset_path)):
        category_path = os.path.join(dataset_path, category)

        # Verificar que la carpeta principal es válida
        if not os.path.isdir(category_path):
            print(f"Saltando {category}: No es un directorio válido.")
            continue

        # Diagnóstico: Mostrar qué categorías se están procesando
        print(f"Procesando categoría: {category}")

        for subcategory in sorted(os.listdir(category_path)):
            subcategory_path = os.path.join(category_path, subcategory)

            # Verificar que el subdirectorio es válido
            if not os.path.isdir(subcategory_path):
                print(f"Saltando {subcategory}: No es un subdirectorio válido.")
                continue

            # Asignar etiquetas según la categoría y subcategoría
            try:
                if category.lower() == "numeros":
                    label = int(subcategory)  # Etiquetas para números (0-9)
                elif category.lower() == "mayusculas":
                    if subcategory == "Ñ":
                        label = 36  # Etiqueta específica para Ñ
                    else:
                        label = 10 + (ord(subcategory.upper()) - ord('A'))  # Etiquetas para A-Z
                elif category.lower() == "minusculas":
                    if subcategory == "ñ":
                        label = 63  # Etiqueta específica para ñ
                    else:
                        label = 37 + (ord(subcategory) - ord('a'))  # Etiquetas para a-z
                else:
                    print(f"Saltando {subcategory}: Categoría no reconocida.")
                    continue
            except ValueError as e:
                print(f"Error al asignar etiqueta para {subcategory}: {e}")
                continue

            # Procesar imágenes dentro del subdirectorio
            for img_name in sorted(os.listdir(subcategory_path)):
                img_path = os.path.join(subcategory_path, img_name)

                # Verificar que el archivo es una imagen
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    print(f"Saltando {img_name}: No es un archivo de imagen.")
                    continue

                try:
                    # Leer, redimensionar y normalizar la imagen
                    img = Image.open(img_path).convert("L")
                    img = img.resize(IMG_SIZE)  # Redimensionar
                    images.append(np.array(img, dtype=np.float32) / 255.0)
                    labels.append(label)
                    print(f"Categoría: {category}, Subcategoría: {subcategory}, Imagen: {img_name}, Etiqueta asignada: {LABEL_MAP[label]}")
                except Exception as e:
                    print(f"Error al procesar la imagen {img_name}: {e}")

    # Convertir las listas a arrays numpy y devolver
    return np.array(images).reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1), np.array(labels)


if __name__ == "__main__":
    # Ruta a la carpeta raíz del dataset
    dataset_path = r"C:\Users\barra\.vscode\Inteligencia Artificial\ExamenFinal\ExamenFinalOCR\DatasetsOCR25-26"

    if not os.path.exists(dataset_path):
        print(f"Ruta del dataset no encontrada: {dataset_path}")
    else:
        images, labels = load_dataset(dataset_path)
        print(f"\n=== Resumen ===")
        print(f"Número total de imágenes cargadas: {len(images)}")
        print(f"Número total de etiquetas cargadas: {len(labels)}")
        print(f"Primeras 10 etiquetas asignadas: {[LABEL_MAP[label] for label in labels[:10]]}")
 





  