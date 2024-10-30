import cv2
import torch
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
from transformers import ViTForImageClassification

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device name: {torch.cuda.get_device_name(0)}')

detector = YOLO('yolov8n.pt')
detector.to(device)

NUM_CLASSES = 12
model_name = 'google/vit-base-patch16-224'

classifier = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=NUM_CLASSES,
)

state_dict = torch.load('trained_model.pth', map_location=device)
classifier.load_state_dict(state_dict)
classifier.to(device)
classifier.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

input_video_path = 'original_video.mp4'
output_video_path = 'demo_video.mp4'


cap = cv2.VideoCapture(input_video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = detector(frame)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = box.conf[0].item() 
            cls = int(box.cls[0].item())
            
            if cls == 0:
                label = f'Person {conf:.2f}'
                
                person_img = frame[y1:y2, x1:x2]
                if person_img.size == 0:
                    continue 

                person_img_pil = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
                input_tensor = transform(person_img_pil).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = classifier(input_tensor)
                    predicted_class = torch.argmax(output.logits, dim=1).item()
                
                label += f', Class: {predicted_class}'
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    out.write(frame)
    
    
cap.release()
out.release()
cv2.destroyAllWindows()
