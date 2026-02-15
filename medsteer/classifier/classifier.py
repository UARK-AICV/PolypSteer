import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from medsteer.classifier.model import KvasirClassifierModule
from medsteer.classifier.dataset import KVASIR_LABELS

class KvasirClassifier(nn.Module):
    def __init__(self, checkpoint_path=None, model_name="densenet121", device="auto", image_size=224):
        super().__init__()
        self.pathologies = KVASIR_LABELS
        self.num_classes = len(KVASIR_LABELS)
        self.image_size = int(image_size)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if checkpoint_path:
            self.model_module = KvasirClassifierModule.load_from_checkpoint(checkpoint_path)
            self.model = self.model_module.model
        else:
            self.model_module = KvasirClassifierModule(model_name=model_name, num_classes=self.num_classes)
            self.model = self.model_module.model

        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)

    def preprocess(self, image: Image.Image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.transform(image).unsqueeze(0)

    @torch.no_grad()
    def classify(self, image: Image.Image):
        tensor = self.preprocess(image).to(self.device)
        probs = self.predict(tensor)[0].cpu().numpy()
        return {label: float(prob) for label, prob in zip(self.pathologies, probs)}

    @torch.no_grad()
    def classify_batch(self, images: list) -> list:
        if not images:
            return []
        tensors = torch.cat([self.preprocess(img) for img in images], dim=0).to(self.device)
        probs = self.predict(tensors).cpu().numpy()

        results = []
        for prob in probs:
            results.append({label: float(p) for label, p in zip(self.pathologies, prob)})
        return results

def load_classifier(checkpoint_path, model_name="densenet121", device="auto", image_size=224):
    return KvasirClassifier(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        device=device,
        image_size=image_size,
    )
