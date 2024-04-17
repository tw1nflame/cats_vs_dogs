import base64
import io
import torch
from torch import nn
from pathlib import Path
from tinyVGG import tinyVGGModel
from torchvision import transforms
from PIL import Image


model_path = Path('models/79acc.pth')
model = tinyVGGModel(3, 10, 1)
classes = ['Cat', 'Dog']
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.to("cpu")

data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


def get_result(image):
    image_bytes = image.file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image_transformed = data_transform(image).to('cpu')
    model.eval()
    with torch.inference_mode():
        predicted_logits = model(image_transformed.unsqueeze(dim=0))
        predicted_class = classes[int(torch.round(
            torch.sigmoid(predicted_logits)).item())]
        encoded_string = base64.b64encode(image_bytes)
        bs64 = encoded_string.decode('utf-8')
        image_data = f'data:image/jpeg;base64,{bs64}'
        return predicted_class, image_data
