import cv2
import torch
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
from transformers import ViTForImageClassification, ViTConfig

# ---------------------------- #
#        1. 加载模型            #
# ---------------------------- #

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 版本: {torch.version.cuda}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
print(f"cuDNN 是否可用: {torch.backends.cudnn.enabled}")


# 初始化设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# 加载 ViT 配置和模型
config = ViTConfig.from_pretrained('best_trained_vit_model/config.json')
classifier = ViTForImageClassification.from_pretrained(
    'best_trained_vit_model',
    config=config,
    trust_remote_code=True  # 根据需要设置
)
classifier.to(device)
classifier.eval()
print("分类器模型已成功加载。")

# 加载 YOLOv8 检测器
detector = YOLO('yolov8n.pt')  # 替换为您的 YOLOv8 权重路径
print("YOLOv8 检测器已成功加载。")

# ---------------------------- #
#        2. 定义预处理          #
# ---------------------------- #

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 根据分类器输入要求调整大小
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 根据训练时的归一化参数
                         std=[0.229, 0.224, 0.225])
])

# ---------------------------- #
#        3. 处理视频            #
# ---------------------------- #

# 输入和输出视频路径
input_video_path = 'your_video.mp4'       # 替换为您的输入视频路径
output_video_path = 'demo_video.mp4'     # 输出视频路径

# 打开输入视频
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print(f"无法打开视频文件 {input_video_path}")
    exit()

# 获取视频信息
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码

# 初始化视频写入器
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
print(f"开始处理视频，输出将保存到 {output_video_path}")

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    print(f"处理第 {frame_count} 帧...", end='\r')

    # 使用 YOLOv8 进行检测
    results = detector(frame)[0]  # 获取第一帧的检测结果

    # 遍历所有检测到的对象
    for box in results.boxes:
        # 获取边界框坐标
        x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])  # 转换为整数
        conf = box.conf.tolist()[0]  # 置信度
        cls = int(box.cls.tolist()[0])  # 类别ID

        # 假设只处理人类（根据您的训练集，调整类别ID）
        # 例如，如果训练集是针对人类的不同类别，则需要根据具体类别ID进行处理
        # 否则，可以根据需要过滤特定类别
        # 这里假设 cls == 0 是人类
        # 根据您的实际情况调整
        # 如果您训练的是针对不同类别的人物，可能不需要过滤
        # 如果您训练的是针对所有类别，请根据需要调整

        # 裁剪边界框内的图像
        person_img = frame[y1:y2, x1:x2]
        if person_img.size == 0:
            continue  # 跳过空图像

        # 转换为 PIL 图像并进行预处理
        person_img_pil = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
        input_tensor = transform(person_img_pil).unsqueeze(0).to(device)  # 添加 batch 维度

        # 分类
        with torch.no_grad():
            outputs = classifier(input_tensor)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            
        class_names = {0: 'Amber', 1: 'Childe', 2: 'Dori', 3: 'Eula', 4: 'Ganyu', 5: 'Hutao',6: 'Lisa', 7: 'Nahida', 8: 'Raiden',
               9: 'Venti', 10: 'Yoimiya', 11: 'Zhongli',}
        label = f'{class_names.get(predicted_class, "Unknown")} ({conf:.2f})'

        # 显示分类结果
        #label = f'Class: {predicted_class} ({conf:.2f})'

        # 绘制边界框
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 在边界框上方显示标签
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 写入帧到输出视频
    out.write(frame)

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
print("\n视频处理完成。")
