import os
import cv2
from ultralytics import YOLO

input_folder = 'C:/Users/jeremy/Downloads/Dori'
output_folder = 'C:/Users\jeremy/Downloads/dori_new/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

model = YOLO('yolov8n.pt')

for image_name in os.listdir(input_folder):
    try:
        image_path = os.path.join(input_folder, image_name)
        image = cv2.imread(image_path)
        results = model(image)
        
        for i, result in enumerate(results):
            for box in result.boxes:
                if box.cls == 0: 
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    person_image = image[y1:y2, x1:x2]
                    save_path = os.path.join(output_folder, f'{image_name}_person_{i}.jpg')
                    cv2.imwrite(save_path, person_image)
        print(f'Processed {image_name}')
        
    except Exception as e:
        print(f'Error processing {image_name}: {e}')
        continue
    
