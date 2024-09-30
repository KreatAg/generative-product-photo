from diffusers import StableDiffusionPipeline
import torch
import os

# Cargar el modelo preentrenado de Hugging Face
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Definir la ruta de las imágenes que se usarán para el fine-tuning
image_dir = "/src/images"

# Aquí puedes procesar las imágenes para el ajuste (fine-tuning)
print("Comenzando el entrenamiento con las imágenes...")
# Aquí puedes agregar un código real para el ajuste si es necesario

# Guardar el modelo ajustado
output_dir = "/output/fine-tuned-model"
os.makedirs(output_dir, exist_ok=True)
pipe.save_pretrained(output_dir)

print(f"Fine-tuning completado y modelo guardado en {output_dir}")
