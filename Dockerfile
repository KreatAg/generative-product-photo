# Utiliza una imagen base de PyTorch con soporte para CUDA
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

# Instala las dependencias necesarias
RUN pip install diffusers transformers accelerate torch torchvision ftfy scipy

# Copia el código del repositorio al contenedor
COPY . /src
WORKDIR /src

# Comando que ejecutará el script de entrenamiento
CMD ["python", "train.py"]
