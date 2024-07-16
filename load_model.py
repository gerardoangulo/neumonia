import torch
from torch import nn

def load_cnn_model(model_path):
    """
    Carga el modelo de red neuronal convolucional desde un archivo.

    Parámetros:
    - model_path: Ruta del archivo del modelo (debe ser un archivo .pth o .pt).

    Retorna:
    - model: Modelo de red neuronal convolucional cargado.
    """
    try:
        # Definición de la arquitectura del modelo (debe coincidir con la del modelo entrenado)
        class WilhemNet86(nn.Module):
            def __init__(self):
                super(WilhemNet86, self).__init__()
                # Definir las capas del modelo aquí
                self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                self.fc1 = nn.Linear(64 * 128 * 128, 128)
                self.fc2 = nn.Linear(128, 2)  # Asumiendo que hay 2 clases

            def forward(self, x):
                x = self.pool(torch.relu(self.conv1(x)))
                x = self.pool(torch.relu(self.conv2(x)))
                x = x.view(-1, 64 * 128 * 128)
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        # Inicialización del modelo
        model = WilhemNet86()
        
        # Carga del modelo entrenado
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()  # Poner el modelo en modo evaluación
        return model
    except Exception as e:
        print(f"Error cargando el modelo: {e}")
        return None

if __name__ == "__main__":
    # Prueba del script con un modelo de ejemplo
    model_path = 'WilhemNet86.pth'  # Asegúrate de que esta sea la ruta correcta a tu archivo de modelo
    model = load_cnn_model(model_path)
    if model is not None:
        print("Modelo cargado exitosamente.")
    else:
        print("Error al cargar el modelo.")
