import pydicom
import numpy as np
import matplotlib.pyplot as plt

def read_dicom(image_path):
    """
    Lee una imagen DICOM y la convierte en un arreglo de numpy.
    
    Parámetros:
    - image_path: Ruta del archivo DICOM.

    Retorna:
    - image_array: Arreglo numpy de la imagen.
    """
    try:
        dicom_image = pydicom.dcmread(image_path)
        image_array = dicom_image.pixel_array
        return image_array
    except Exception as e:
        print(f"Error leyendo el archivo DICOM: {e}")
        return None

def show_image(image_array):
    """
    Muestra la imagen usando matplotlib.

    Parámetros:
    - image_array: Arreglo numpy de la imagen.
    """
    plt.imshow(image_array, cmap='gray')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Prueba del script con una imagen de ejemplo
    example_image_path = 'example.dcm'
    image_array = read_dicom(example_image_path)
    if image_array is not None:
        show_image(image_array)
