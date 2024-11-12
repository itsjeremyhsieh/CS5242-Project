import cv2
import torch
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
from transformers import ViTForImageClassification, ViTConfig


print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"cuDNN available: {torch.backends.cudnn.enabled}")

# initialization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# loat ViT parameters and model
config = ViTConfig.from_pretrained('best_trained_vit_model/config.json')
classifier = ViTForImageClassification.from_pretrained(
    'best_trained_vit_model',
    config=config,
    trust_remote_code=True  
)
classifier.to(device)
classifier.eval()

# load YOLOv8
detector = YOLO('yolov8n.pt')  

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                         std=[0.229, 0.224, 0.225])
])

# video path
input_video_path = 'your_video.mp4'      
output_video_path = 'demo_video.mp4'     

cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print(f"Cannot load {input_video_path}")
    exit()

# fetch frame
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = detector(frame)[0]  # get first frame information

    for box in results.boxes:
        # get bounding box
        x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0]) 
        conf = box.conf.tolist()[0]  
        cls = int(box.cls.tolist()[0]) 

        person_img = frame[y1:y2, x1:x2]
        if person_img.size == 0:
            continue  

        person_img_pil = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
        input_tensor = transform(person_img_pil).unsqueeze(0).to(device)  
        
        # classification
        with torch.no_grad():
            outputs = classifier(input_tensor)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            
        label = predicted_class

        # draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # add label
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

