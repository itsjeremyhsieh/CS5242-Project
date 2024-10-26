import os
import albumentations as A
import cv2

augmentation_pipeline = A.Compose([

    A.Perspective(scale=(0.05, 0.1), p=0.5),

    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.0, rotate_limit=45, p=0.5),

    A.RandomScale(scale_limit=0.4, p=0.5),

    A.MotionBlur(blur_limit=5, p=0.5),

    A.GaussNoise(var_limit=(50.0, 150.0), p=0.5),

    A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, p=0.5),
    
    A.RandomSunFlare(flare_roi=(0, 0.5, 1, 1), angle_lower=0, angle_upper=1, num_flare_circles_lower=1, num_flare_circles_upper=2, src_radius=200, p=0.5),
    
    A.CoarseDropout(max_holes=2, max_height=50, max_width=50, min_holes=1, min_height=20, min_width=20, fill_value=0, p=0.5)
])


def augment_image(image):
    augmented = augmentation_pipeline(image=image)
    return augmented['image']

def process_folder(classes, num_aug=5):
    
    for char in classes:
        input_folder = f"extracted/{char}_extracted/" # remember to change this!!!
        output_folder = f"augmented/{char}_augmented/"
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    image_path = os.path.join(input_folder, filename)
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    for i in range(num_aug): # augment the image num_aug times
                        augmented_image = augment_image(image)
                        augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
                        
                        output_filename = f"{os.path.splitext(filename)[0]}_aug_{i+1}.jpg" # save the image
                        output_path = os.path.join(output_folder, output_filename)
                        cv2.imwrite(output_path, augmented_image)
                        
                    print(f"Processed {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    continue
        
# process_folder(['yoimiya'])