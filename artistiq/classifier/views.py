from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import os
import torch
from torchvision import transforms
from PIL import Image
from .model import CNN_basic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = ["circle","hexagon","square","star","triangle"]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "cnn_basic.pt")

model = CNN_basic(num_classes=5).to(device)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28,28)),
])

@csrf_exempt
def predict_shape(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=400)

    if "image" not in request.FILES:
        return JsonResponse({"error": "No image uploaded"}, status=400)

    img_file = request.FILES["image"]

    save_path = os.path.join(settings.MEDIA_ROOT, "input.jpg")
    with open(save_path, "wb+") as f:
        for chunk in img_file.chunks():
            f.write(chunk)

    img = Image.open(save_path).convert("RGB")

    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = model.return_probs()
        index = torch.argmax(output).item()
        prediction = CLASSES[index]

    return JsonResponse({"prediction": prediction, "score": round(torch.max(probs).item()*100) })
