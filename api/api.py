from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
from torchvision import transforms
from PIL import Image
import io
import logging
import sys

sys.path.append("/Users/paulchauvin/Documents/GitHub/lunit/src/")
from config_loader import load_config
from train import train_model
from models import SimpleConvNet
app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model on startup
MODEL_PATH = "mlruns/0/e7755442900b447d8438f1466523d07a/artifacts/models/model_state_dict.pth" #select best model for inference
model = SimpleConvNet()

state = torch.load(MODEL_PATH)
print(type(state))


model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

@app.post("/inference/")
async def run_inference(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("L")
        image = transform(image)
        image = image.unsqueeze(0)

        with torch.no_grad():
            output = model(image)
            prediction = output.argmax().item()

        return {"prediction": prediction}
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail="Inference error")

@app.post("/train/")
async def run_train(config_file: UploadFile = File(...)):
    try:
        temp_config_path = "temp_config.yaml"
        with open(temp_config_path, "wb") as buffer:
            buffer.write(config_file.file.read())
        train_model(temp_config_path)
        return {"status": "Training started with the provided configuration."}
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise HTTPException(status_code=500, detail="Training error")
