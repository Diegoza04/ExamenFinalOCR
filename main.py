import os
import zipfile
import pandas as pd
import numpy as np
import cv2
import sys
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

model_save_path = "ocr_model.h5"


# Aseguramos que el sistema utilice UTF-8 como codificación predeterminada
print("Default encoding:", sys.getdefaultencoding())
os.environ["PYTHONIOENCODING"] = "utf-8"  # Garantizamos que las operaciones usen UTF-8

# 1. Extract the dataset from a local zip file
print("Extracting dataset from local zip file...")

# Local paths for the zip file and dataset extraction
zip_path = "C:\\Users\\barra\\Downloads\\DatasetsOCR25-26IA.zip"
extract_path = "C:\\Users\\barra\\Documents\\DatasetsOCR25-26"

# Unzip the dataset
if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
print("Dataset extracted successfully.")


# 2. Create the df_images DataFrame
print("Creating df_images DataFrame...")
image_paths = []
character_types = []
character_labels = []

# Almacenar y procesar imágenes
for root, dirs, files in os.walk(extract_path):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(root, file)

            try:
                # Intentar leer la imagen para verificar su validez
                print(f"Verificando: {image_path}")
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError(f"No se pudo leer la imagen: {image_path}")

                # Guardar la ruta si pasa la verificación
                image_paths.append(image_path)

                # Extraer tipo de carácter y etiqueta (asumiendo estructura .../character_type/character_label/file)
                parts = image_path.split(os.sep)
                if len(parts) >= 3:  # Asegurarse de que haya suficientes partes para extraer info
                    character_label = parts[-2]
                    character_type = parts[-3]
                    character_types.append(character_type)
                    character_labels.append(character_label)
                else:
                    character_types.append(None)
                    character_labels.append(None)
            except Exception as e:
                print(f"Error al procesar {image_path}: {e}")

df_images = pd.DataFrame({
    'image_path': image_paths,
    'character_type': character_types,
    'character_label': character_labels
})

# Filtrar filas donde no se pudo determinar la etiqueta o el tipo
df_images.dropna(subset=['character_label'], inplace=True)
print(f"df_images created with {len(df_images)} images.")

# 3. Define the preprocess_image_cnn function (for paths)
def preprocess_image_cnn(image_path, target_size=(64, 64)):
    """
    Loads an image, converts it to grayscale, resizes it, and normalizes pixel values to [0, 1].

    Args:
        image_path (str): The path to the image file.
        target_size (tuple): A tuple (width, height) for the target size.

    Returns:
        numpy.ndarray: The preprocessed, normalized image (grayscale, resized, float type).
        None: If the image cannot be loaded.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not load image from {image_path}")
        return None

    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize image
    resized_img = cv2.resize(gray_img, target_size, interpolation=cv2.INTER_AREA)

    # Normalize pixel values to [0, 1]
    normalized_img = resized_img.astype(np.float32) / 255.0

    return normalized_img

# Define the process_char_for_model_cnn function (for image arrays)
def process_char_for_model_cnn(char_roi_array, target_size=(64, 64)):
    """
    Resizes a grayscale character image ROI for CNN model input, normalizes it, and adds a channel dimension.

    Args:
        char_roi_array (numpy.ndarray): The grayscale NumPy array of the character ROI.
        target_size (tuple): A tuple (width, height) for the target size.

    Returns:
        numpy.ndarray: The preprocessed, normalized, and reshaped character image array (height, width, 1).
        None: If the input array is invalid.
    """
    # Ensure the input is not None and is a valid image array
    if char_roi_array is None or char_roi_array.size == 0:
        return None

    # Resize image
    resized_img = cv2.resize(char_roi_array, target_size, interpolation=cv2.INTER_AREA)

    # Normalize pixel values to [0, 1]
    normalized_img = resized_img.astype(np.float32) / 255.0

    # Add channel dimension (for grayscale, it's 1 channel)
    final_img = np.expand_dims(normalized_img, axis=-1)

    return final_img

print("Preprocessing functions defined.")

# 4. Apply preprocess_image_cnn and create df_filtered_cnn
print("Applying preprocess_image_cnn to df_images...")
df_images['preprocessed_cnn_image'] = df_images['image_path'].apply(preprocess_image_cnn)
df_filtered_cnn = df_images[df_images['preprocessed_cnn_image'].notna()].copy()
print(f"df_filtered_cnn created with {len(df_filtered_cnn)} valid preprocessed images.")


# 5. Convert to NumPy arrays (X_cnn, y_cnn_one_hot)
print("Converting preprocessed images and labels to NumPy arrays...")
X_cnn = np.array(df_filtered_cnn['preprocessed_cnn_image'].tolist())
X_cnn = X_cnn.reshape(X_cnn.shape[0], X_cnn.shape[1], X_cnn.shape[2], 1)
y_cnn = df_filtered_cnn['character_label'].values

label_encoder = LabelEncoder()
y_cnn_encoded = label_encoder.fit_transform(y_cnn)
num_classes = len(label_encoder.classes_)
y_cnn_one_hot = to_categorical(y_cnn_encoded, num_classes=num_classes)
print(f"X_cnn shape: {X_cnn.shape}, y_cnn_one_hot shape: {y_cnn_one_hot.shape}")

# 6. Split data into training/testing sets
print("Splitting data into training and testing sets...")
X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(
    X_cnn, y_cnn_one_hot, test_size=0.2, random_state=42, stratify=y_cnn_encoded
)
print("Data split successfully.")

# 7. Define model_cnn with adjusted architecture
# Variable global para marcar si el modelo fue cargado o entrenado
model_loaded = False

# Verificar si el modelo ya existe
if os.path.exists(model_save_path):
    print(f"Cargando el modelo guardado desde: {model_save_path}")
    model_cnn = load_model(model_save_path)
    print("Modelo cargado exitosamente.")
    model_loaded = True
else:
    print("No se encontró un modelo guardado. Entrenaremos el modelo desde cero.")
if not model_loaded:
    print("Defining CNN model with adjusted architecture...")
    model_cnn = Sequential()
    model_cnn.add(Conv2D(64, (5, 5), activation='relu', input_shape=X_cnn.shape[1:]))
    model_cnn.add(MaxPooling2D((2, 2)))
    model_cnn.add(Conv2D(128, (3, 3), activation='relu'))
    model_cnn.add(MaxPooling2D((2, 2)))
    model_cnn.add(Conv2D(256, (3, 3), activation='relu'))
    model_cnn.add(MaxPooling2D((2, 2)))
    model_cnn.add(Flatten())
    model_cnn.add(Dense(256, activation='relu'))
    model_cnn.add(Dropout(0.6))
    model_cnn.add(Dense(num_classes, activation='softmax'))
    model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model_cnn.summary()
    print("CNN model defined and compiled.")

    # 8. Train the model_cnn
    print("Training CNN model...")
    history_new_arch = model_cnn.fit(
        X_train_cnn, y_train_cnn,
        epochs=30,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    print("Entrenamiento completado.")

        # Guardar el modelo
    print(f"Guardando el modelo en: {model_save_path}")
    model_cnn.save(model_save_path)
    print("Modelo guardado exitosamente.")

# 9. Define the ocr_pipeline function with segmentation refinements
def ocr_pipeline(image_path):
    """
    OCR pipeline to recognize text from an image using the trained CNN model.

    Args:
        image_path (str): Path to the input image.

    Returns:
        tuple: (str, numpy.ndarray) - recognized text and image with bounding boxes.
    """
    original_img_bgr = cv2.imread(image_path)
    if original_img_bgr is None:
        print(f"Error: Could not load image from {image_path}")
        return "", None

    original_img_rgb = cv2.cvtColor(original_img_bgr, cv2.COLOR_BGR2RGB)
    img_with_boxes = original_img_rgb.copy()

    gray_img = cv2.cvtColor(original_img_bgr, cv2.COLOR_BGR2GRAY)
    binary_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Apply dilation to connect fragmented character components
    kernel = np.ones((3,3), np.uint8)
    binary_img = cv2.dilate(binary_img, kernel, iterations=1)

    horizontal_projection = np.sum(binary_img, axis=1)
    line_regions = []
    in_text_region = False
    start_row = 0
    threshold_projection = np.max(horizontal_projection) * 0.05

    for i, intensity in enumerate(horizontal_projection):
        if intensity > threshold_projection and not in_text_region:
            start_row = i
            in_text_region = True
        elif intensity <= threshold_projection and in_text_region:
            end_row = i
            if end_row - start_row > 5:
                line_regions.append((start_row, end_row))
            in_text_region = False
    if in_text_region:
        line_regions.append((start_row, len(horizontal_projection) - 1))

    recognized_text_lines = []

    for line_idx, (start_row, end_row) in enumerate(line_regions):
        line_binary_img = binary_img[start_row:end_row, :]
        line_gray_img = gray_img[start_row:end_row, :]

        vertical_projection = np.sum(line_binary_img, axis=0)

        char_regions = []
        in_char_region = False
        start_col = 0
        threshold_vertical_projection = np.max(vertical_projection) * 0.05

        for i, intensity in enumerate(vertical_projection):
            if intensity > threshold_vertical_projection and not in_char_region:
                start_col = i
                in_char_region = True
            elif intensity <= threshold_vertical_projection and in_char_region:
                end_col = i
                if end_col - start_col > 3:
                    char_regions.append((start_col, end_col))
                in_char_region = False
        if in_char_region:
            char_regions.append((start_col, len(vertical_projection) - 1))

        current_line_text = ""

        for char_start_col, char_end_col in char_regions:
            char_roi = line_gray_img[:, char_start_col:char_end_col]
            # FIX: Call process_char_for_model_cnn instead of preprocess_image_cnn
            char_features_cnn = process_char_for_model_cnn(char_roi)

            if char_features_cnn is not None and char_features_cnn.size > 0:
                input_for_prediction = np.expand_dims(char_features_cnn, axis=0)
                predictions = model_cnn.predict(input_for_prediction, verbose=0)
                predicted_class_index = np.argmax(predictions, axis=1)[0]
                predicted_char = label_encoder.inverse_transform([predicted_class_index])[0]
                current_line_text += predicted_char

                cv2.rectangle(img_with_boxes,
                              (char_start_col, start_row),
                              (char_end_col, end_row),
                              (0, 255, 0), 2)
            else:
                current_line_text += '?'

        recognized_text_lines.append(current_line_text)

        cv2.rectangle(img_with_boxes,
                      (0, start_row),
                      (img_with_boxes.shape[1], end_row),
                      (255, 0, 0), 2)

    final_recognized_text = " ".join(recognized_text_lines)

    return final_recognized_text, img_with_boxes

print("OCR pipeline function with refined segmentation defined.")

# 10. Select one image for each unique character
print("Selecting one image for each unique character...")
sampled_unique_char_images_info = []
unique_char_labels = label_encoder.classes_

for char_label in unique_char_labels:
    char_df = df_filtered_cnn[df_filtered_cnn['character_label'] == char_label]
    if not char_df.empty:
        sample_row = char_df.iloc[0]
        sampled_unique_char_images_info.append({
            'image_path': sample_row['image_path'],
            'character_label': sample_row['character_label']
        })
sampled_unique_char_images_df = pd.DataFrame(sampled_unique_char_images_info)
print(f"Selected {len(sampled_unique_char_images_df)} unique character sample images.")

# 11. Iterate through sampled_unique_char_images_df and store results
print("Evaluating OCR pipeline on unique character samples...")
evaluation_results_after_arch_adj = []

for index, row in sampled_unique_char_images_df.iterrows():
    image_path = row['image_path']
    true_label = row['character_label']

    recognized_text_cnn, _ = ocr_pipeline(image_path)

    is_correct = (recognized_text_cnn == true_label)

    evaluation_results_after_arch_adj.append({
        'image_path': image_path,
        'true_label': true_label,
        'recognized_text_cnn': recognized_text_cnn,
        'is_correct': is_correct
    })
ocr_char_evaluation_df_after_arch_adj = pd.DataFrame(evaluation_results_after_arch_adj)
print("Evaluation completed and results stored.")

# 12. Calculate and print overall accuracy
print("Calculating overall accuracy...")
total_chars_after_arch_adj = len(ocr_char_evaluation_df_after_arch_adj)
correct_recognitions_after_arch_adj = ocr_char_evaluation_df_after_arch_adj['is_correct'].sum()
overall_accuracy_after_arch_adj = (correct_recognitions_after_arch_adj / total_chars_after_arch_adj) * 100

print(f"\n--- Summary of OCR Pipeline Evaluation by Unique Character (After Architecture Adjustments) ---")
print(f"Total unique characters evaluated: {total_chars_after_arch_adj}")
print(f"Number of correct recognitions: {correct_recognitions_after_arch_adj}")
print(f"Overall accuracy on unique character samples: {overall_accuracy_after_arch_adj:.2f}%")

# 13. Print correctly and incorrectly recognized characters
correctly_recognized_chars_after_arch_adj = ocr_char_evaluation_df_after_arch_adj[
    ocr_char_evaluation_df_after_arch_adj['is_correct'] == True]['true_label'].tolist()
incorrectly_recognized_info_after_arch_adj = ocr_char_evaluation_df_after_arch_adj[
    ocr_char_evaluation_df_after_arch_adj['is_correct'] == False][['true_label', 'recognized_text_cnn']].values.tolist()

print("\nCharacters correctly recognized:")
if correctly_recognized_chars_after_arch_adj:
    print(correctly_recognized_chars_after_arch_adj)
else:
    print("None")

print("\nCharacters incorrectly recognized (True vs. Recognized):")
if incorrectly_recognized_info_after_arch_adj:
    for true, recognized in incorrectly_recognized_info_after_arch_adj:
        print(f"  True: '{true}', Recognized: '{recognized}'")
else:
    print("None")
print("Evaluation summary generated.")

# 14. Visualize evaluation results
print("Visualizing OCR pipeline results for each unique character...")
num_images_after_arch_adj = len(ocr_char_evaluation_df_after_arch_adj)
n_cols_after_arch_adj = 8
n_rows_after_arch_adj = (num_images_after_arch_adj + n_cols_after_arch_adj - 1) // n_cols_after_arch_adj

plt.figure(figsize=(n_cols_after_arch_adj * 3, n_rows_after_arch_adj * 3))

for i, row in ocr_char_evaluation_df_after_arch_adj.iterrows():
    image_path = row['image_path']
    true_label = row['true_label']
    recognized_text = row['recognized_text_cnn']

    _, annotated_image = ocr_pipeline(image_path)

    if annotated_image is None:
        print(f"Warning: Could not get annotated image for {image_path}. Skipping.")
        continue

    plt.subplot(n_rows_after_arch_adj, n_cols_after_arch_adj, i + 1)
    plt.imshow(annotated_image)
    plt.title(f"True: '{true_label}', Rec: '{recognized_text}'")
    plt.axis('off')

plt.tight_layout()
plt.show()
print("Visualization of OCR pipeline results completed.")


# 16. Interfaz gráfica para seleccionar una imagen y probar el modelo
def select_and_test_image():
    """
    Muestra un cuadro de diálogo para que el usuario seleccione una imagen
    y prueba el modelo en la imagen seleccionada.
    """
    # Crear la ventana emergente de selección de archivo
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal

    messagebox.showinfo("Selecciona una imagen", "Selecciona la imagen que deseas probar con el modelo.")

    # Abrir un cuadro de diálogo para que el usuario seleccione el archivo
    file_path = filedialog.askopenfilename(
        title="Seleccionar imagen",
        filetypes=[("Imagenes", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"), ("Todos los archivos", "*.*")]
    )

    # Comprobar si el usuario seleccionó algo
    if not file_path:
        messagebox.showwarning("No se seleccionó ninguna imagen", "Por favor, selecciona una imagen válida.")
        return

    # Procesar la imagen seleccionada
    print(f"Procesando la imagen seleccionada: {file_path}")
    recognized_text, annotated_image = ocr_pipeline(file_path)

    # Mostrar resultados
    if annotated_image is not None:
        plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Texto reconocido: '{recognized_text}'")
        plt.axis('off')
        plt.show()
    else:
        messagebox.showerror("Error al procesar la imagen", "No se pudo procesar la imagen correctamente.")

# Crear una segunda función para ejecutar la ventana gráfica global
def open_gui():
    """
    Crea una ventana gráfica con un botón para seleccionar y procesar una imagen.
    """
    # Crear la ventana principal
    gui_root = tk.Tk()
    gui_root.title("Probar modelo OCR con imagen")

    # Configurar el tamaño de la ventana
    gui_root.geometry("400x200")

    # Etiqueta instructiva
    label = tk.Label(gui_root, text="Pulsa el botón para seleccionar una imagen:")
    label.pack(pady=20)

    # Botón para seleccionar y procesar una imagen
    select_button = tk.Button(gui_root, text="Seleccionar imagen", command=select_and_test_image)
    select_button.pack(pady=10)

    # Botón de salida
    quit_button = tk.Button(gui_root, text="Salir", command=gui_root.destroy)
    quit_button.pack(pady=10)

    # Ejecutar el bucle principal de la GUI
    gui_root.mainloop()

# Llamar directamente a la función para abrir la GUI
if __name__ == "__main__":
    open_gui()
