import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from load_model import load_cnn_model

def preprocess_image(image_path):
    # Preprocesamiento básico para la imagen
    image = Image.open(image_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image = transform(image).unsqueeze(0)
    return image

def generate_grad_cam(model, image_tensor, target_layer_name):
    """
    Genera el mapa de calor Grad-CAM para una imagen dada.

    Parámetros:
    - model: Modelo de red neuronal convolucional.
    - image_tensor: Tensor de la imagen preprocesada.
    - target_layer_name: Nombre de la capa convolucional objetivo para Grad-CAM.

    Retorna:
    - class_name: Nombre de la clase predicha.
    - probability: Probabilidad de la clase predicha.
    - heatmap: Mapa de calor generado.
    """
    def forward_hook(module, input, output):
        """
        Hook para capturar la salida de la capa objetivo.
        """
        global feature_maps
        feature_maps = output

    def backward_hook(module, grad_in, grad_out):
        """
        Hook para capturar el gradiente de la capa objetivo.
        """
        global gradients
        gradients = grad_out[0]

    # Obtener la capa objetivo
    target_layer = dict(model.named_modules())[target_layer_name]

    # Registrar los hooks
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)

    # Hacer la predicción
    output = model(image_tensor)
    pred_class = output.argmax(dim=1).item()
    probability = F.softmax(output, dim=1)[0][pred_class].item()
    
    # Clase predicha
    class_name = f"Class {pred_class}"

    # Calcular el gradiente de la clase objetivo con respecto a la salida
    model.zero_grad()
    class_score = output[0][pred_class]
    class_score.backward()

    # Obtener los gradientes y los feature maps
    global gradients, feature_maps
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(feature_maps.shape[1]):
        feature_maps[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(feature_maps, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)

    # Convertir el heatmap a un formato adecuado para visualización
    heatmap = heatmap.detach().cpu().numpy()
    heatmap = cv2.resize(heatmap, (512, 512))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superponer el heatmap en la imagen original
    image = image_tensor.squeeze().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    superimposed_img = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    # Desregistrar los hooks
    handle_forward.remove()
    handle_backward.remove()

    # Guardar el heatmap
    heatmap_path = 'heatmap.png'
    cv2.imwrite(heatmap_path, superimposed_img)

    return class_name, probability, heatmap_path

if __name__ == "__main__":
    # Prueba del script con una imagen de ejemplo
    model_path = 'WilhemNet86.pth'
    image_path = 'example.dcm'
    
    model = load_cnn_model(model_path)
    image_tensor = preprocess_image(image_path)
    target_layer_name = 'conv2'  # Nombre de la capa objetivo

    class_name, probability, heatmap_path = generate_grad_cam(model, image_tensor, target_layer_name)
    print(f"Prediction: {class_name}, Probability: {probability}, Heatmap saved at: {heatmap_path}")

    # Mostrar el heatmap
    heatmap = cv2.imread(heatmap_path)
    plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
