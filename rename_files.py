import os

def rename_files(directory):
    for foldername, subfolders, filenames in os.walk(directory):
        for filename in filenames:
            new_filename = filename.encode('ascii', 'ignore').decode('utf-8')  # Elimina caracteres especiales
            if filename != new_filename:
                old_path = os.path.join(foldername, filename)
                new_path = os.path.join(foldername, new_filename)
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")

# Ruta al directorio raíz del dataset
dataset_directory = "C:/Users/barra/Documents/DatasetsOCR25-26"
rename_files(dataset_directory)