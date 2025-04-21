import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image

class Config:
    # Настройки данных
    DATA_PATH = "dataset"                     # Путь к dataset
    CLASSES = ["Ара", "Какаду", "Амазон"]     # Список классов
    IMG_SIZE = 224                            # Размер входного изображения
    
    # Настройки детекции (только для инференса)
    USE_DETECTION = True                      # Использовать детектор
    DETECTOR_TYPE = "yolo"                    # yolo/detr
    DETECTION_THRESHOLD = 0.7                 # Порог уверенности
    
    # Настройки модели
    ARCHITECTURE = "AlexNet"                  # AlexNet
    PRETRAINED = False                        # Использовать предобученные веса
    SAVE_PATH = "parrot_net.pth"              # Путь для сохранения модели
    
    # Параметры обучения
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 0.001
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Реализация AlexNet-style сети
class ParrotNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, len(Config.CLASSES)),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256*6*6)
        return self.classifier(x)

# Аугментации
train_transform = transforms.Compose([
    transforms.Resize(Config.IMG_SIZE + 32),
    transforms.RandomResizedCrop(Config.IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Загрузка данных
def load_data():
    dataset = datasets.ImageFolder(
        Config.DATA_PATH,
        transform=train_transform
    )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    return random_split(dataset, [train_size, val_size])

# Инициализация модели
def init_model():
    if Config.ARCHITECTURE == "AlexNet":
        model = ParrotNet()
    return model.to(Config.DEVICE)

# Обучение
def train():
    train_set, val_set = load_data()
    
    model = init_model()
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    
    for epoch in range(Config.EPOCHS):
        # Training phase
        model.train()
        for inputs, labels in DataLoader(train_set, batch_size=Config.BATCH_SIZE):
            inputs = inputs.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation phase
        model.eval()
        correct = 0
        with torch.no_grad():
            for inputs, labels in DataLoader(val_set, batch_size=Config.BATCH_SIZE):
                inputs = inputs.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
        
        val_acc = correct / len(val_set)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), Config.SAVE_PATH)
        
        print(f"Epoch {epoch+1}/{Config.EPOCHS} | Val Acc: {val_acc:.4f}")
    
    print(f"Best Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    train()