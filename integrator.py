import read_img
import preprocess_img
import load_model
import grad_cam

def process_image(image_path):
    # Leer la imagen DICOM
    dicom_image = read_img.read_dicom(image_path)
    
    # Preprocesar la imagen
    preprocessed_image = preprocess_img.preprocess(dicom_image)
    
    # Cargar el modelo
    model = load_model.load_cnn_model('WilhemNet86.h5')
    
    # Obtener la predicción y el mapa de calor
    class_name, probability, heatmap = grad_cam.generate_grad_cam(model, preprocessed_image)
    
    # Guardar el mapa de calor en un archivo
    heatmap_path = 'heatmap.png'
    heatmap.save(heatmap_path)
    
    return class_name, probability, heatmap_path

if __name__ == "__main__":
    # Prueba del script con una imagen de ejemplo
    example_image_path = 'example.dcm'
    class_name, probability, heatmap_path = process_image(example_image_path)
    print(f"Prediction: {class_name}, Probability: {probability}, Heatmap saved at: {heatmap_path}")
