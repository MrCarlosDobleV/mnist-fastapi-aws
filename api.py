from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

class ImageInput(BaseModel):
    image: list  # 28x28 list, normalized

app = FastAPI()

model = MNISTNet()
model.load_state_dict(torch.load("mnist_model.pt", map_location="cpu"))
model.eval()

@app.post("/predict")
def predict(data: ImageInput):
    x = torch.tensor(data.image).float().view(1, 1, 28, 28)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        confidence, prediction = torch.max(probs, dim=1)

    return {
        "prediction": int(prediction.item()),
        "confidence": float(confidence.item())
    }


@app.get("/health")
def health():
    return {"status": "ok"}
