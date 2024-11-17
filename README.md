# NUS CS5242 Group 5 Final Project
## Detection and Classification of Characters in Cosplay Field

### Group members
- Hsieh Yu Hsuan, A0304684B
- Li Xu, A0304365J
- Yang Ziqin, A0304993W
- Victor Min Thura Shwe, A0304895U

### Introduction
 This project introduces a Vision Transformer (ViT)-based method for recognizing 12 specific characters from the game Genshin Impact in cosplay images. 
 
 ### Steps
 1. Data collection: for each character,  collect a dataset by web-scraping 150 images.
 2. Data preprocessing:
    - First, for all collected images, perform image cropping by running `data_cropping/crop_person.py`. 
    - Next, manually check for the validation of the cropped image.
    - Finally, we perform data augmentation by running `data_augmentation/augmentation.py`. The code will process 3 augmented images for each input image.
3. Fine-tuning hyperparameters:
    - find the optimal hyperparameters using each corresponding python script.
4. Pipeline inference:
    - to inference on image, run `image_inference.ipynb`
    - to inference on video, run `all_inference.py`

Note: for each specific folder, there is also a `requirements.txt` file. Specified packages must be installed first in order to successfully execute the script. Install the script by running `pip install -r requirements.txt`
Note: dataset should be located at the `root` directory. To perform inference, model weight should be placed under the `pipeline_inference` folder

### Directory Structure

```python
CS5242-PROJECT
├── data_preprocessing/
│   ├── data_augmentation/
│   ├── data_cropping/
├── fine_tuning/
│   ├── activate_function_tuning/
│   ├── learning_rate_tuning/
│   ├── loss_function_tuning
│   ├── optimizer_tuning
├── pipeline_inference/
│   ├── all_inference.py
│   ├── image_inference.ipynb
├── transformer/
│   ├── Character_recognition_Vit.ipynb/
│   ├── requirements.txt
```
