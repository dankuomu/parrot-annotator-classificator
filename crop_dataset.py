import os
import torch
from PIL import Image
from pathlib import Path
import argparse
from tqdm import tqdm

# Конфигурация
class Config:
    yolo_output_dir = "dataset_yolo_cropped"
    detr_output_dir = "dataset_detr_cropped"
    detection_classes = ["bird"]  # Классы для детекции
    yolo_model_name = 'yolov5s'
    confidence_threshold = 0.5

def setup_directories(base_path):
    """Создание директорий для сохранения результатов"""
    Path(Config.yolo_output_dir).mkdir(exist_ok=True)
    Path(Config.detr_output_dir).mkdir(exist_ok=True)
    
    # Создаем поддиректории для классов
    for root, dirs, files in os.walk(base_path):
        if dirs:
            for dir_name in dirs:
                Path(os.path.join(Config.yolo_output_dir, dir_name)).mkdir(exist_ok=True)
                Path(os.path.join(Config.detr_output_dir, dir_name)).mkdir(exist_ok=True)

def load_models():
    """Загрузка моделей детекции"""
    # YOLO
    yolo_model = torch.hub.load('ultralytics/yolov5', Config.yolo_model_name, pretrained=True)
    
    # DETR
    from transformers import DetrImageProcessor, DetrForObjectDetection
    detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    
    return yolo_model, detr_model, detr_processor

def detect_and_crop(image, model, processor=None, model_type='yolo'):
    """Детекция и обрезка изображения"""
    if model_type == 'yolo':
        results = model([image])
        detections = results.pandas().xyxy[0]
        filtered = detections[
            (detections['name'].isin(Config.detection_classes)) &
            (detections['confidence'] > Config.confidence_threshold)
        ]
        
    elif model_type == 'detr':
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=Config.confidence_threshold
        )[0]
        
        filtered = [
            (box, score, label) 
            for box, score, label in zip(results["boxes"], results["scores"], results["labels"]) 
            if model.model.config.id2label[label.item()] in Config.detection_classes
        ]
    
    if len(filtered) == 0:
        return None
    
    # Выбираем бокс с максимальным confidence
    if model_type == 'yolo':
        best = filtered.iloc[0]
        bbox = best[['xmin', 'ymin', 'xmax', 'ymax']].values.astype(int)
    else:
        best_idx = torch.argmax(results["scores"])
        bbox = results["boxes"][best_idx].int().tolist()
    
    # Обрезаем изображение
    return image.crop(bbox)

def process_dataset(dataset_path):
    """Обработка всего датасета"""
    yolo_model, detr_model, detr_processor = load_models()
    
    for root, dirs, files in os.walk(dataset_path):
        for file in tqdm(files, desc=f"Processing {os.path.basename(root)}"):
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                try:
                    img_path = os.path.join(root, file)
                    image = Image.open(img_path).convert('RGB')
                    
                    # Определяем относительный путь для сохранения
                    rel_path = os.path.relpath(root, dataset_path)
                    
                    # Обработка YOLO
                    yolo_crop = detect_and_crop(image, yolo_model, model_type='yolo')
                    if yolo_crop:
                        save_path = os.path.join(Config.yolo_output_dir, rel_path, file)
                        yolo_crop.save(save_path)
                    
                    # Обработка DETR
                    detr_crop = detect_and_crop(image, detr_model, detr_processor, 'detr')
                    if detr_crop:
                        save_path = os.path.join(Config.detr_output_dir, rel_path, file)
                        detr_crop.save(save_path)
                        
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset cropping with YOLO and DETR')
    parser.add_argument('dataset_path', type=str, help='Path to original dataset')
    args = parser.parse_args()
    
    setup_directories(args.dataset_path)
    process_dataset(args.dataset_path)
    print("Dataset processing completed!")