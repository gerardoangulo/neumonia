import cv2
import numpy as np
import torch

def preprocess(image_array):
    """
    Preprocesa la imagen para el modelo de red neuronal.

    Parámetros:
    - image_array: Arreglo numpy de la imagen.

    Retorna:
    - image_tensor: Imagen preprocesada en formato tensor.
    """
    try:
        # Resize a 512x512
        image_resized = cv2.resize(image_array, (512, 512))
        
        # Conversión a escala de grises (aunque ya debería estar en gris, aseguramos)
        if len(image_resized.shape) == 3 and image_resized.shape[2] == 3:
            image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image_resized

        # Ecualización del histograma con CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image_clahe = clahe.apply(image_gray)

        # Normalización de la imagen entre 0 y 1
        image_normalized = image_clahe / 255.0

        # Conversión del arreglo de imagen a formato de batch (tensor)
        image_tensor = torch.tensor(image_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        return image_tensor
    except Exception as e:
        print(f"Error en el preprocesamiento de la imagen: {e}")
        return None

if __name__ == "__main__":
    # Prueba del script con una imagen de ejemplo
    import read_img

    example_image_path = 'example.dcm'
    image_array = read_img.read_dicom(example_image_path)
    
    if image_array is not None:
        image_tensor = preprocess(image_array)
        print(f"Imagen preprocesada: {image_tensor.shape}")
