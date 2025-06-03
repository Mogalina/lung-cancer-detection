# Early Lung Cancer Detection Using Artificial Intelligence

This project presents a modular AI pipeline for the early detection and classification of lung cancer from chest CT scans. By combining deep learning and medical imaging, it aims to support radiologists in diagnosing lung abnormalities earlier and more accurately.

---

## Project Overview

This system uses a two-stage deep learning approach:

1. Classification Model \
   A ResNet50-based CNN classifies each CT image as:
   - Normal
   - Benign
   - Malignant

3. Localization Model \
   If the image is benign or malignant, a secondary model outputs bounding boxes highlighting the tumor region.

### Features:
- Dataset: IQ-OTH/NCCD (TCIA)
- Tools: Python, PyTorch
- Architecture: Fine-tuned ResNet50
- Evaluation Metrics: Accuracy, AUC, Sensitivity, Specificity

---

## Team Members

- Eric Moghioros ([GitHub](https://github.com/Mogalina))
- Andreea Bianca Croitoru ([GitHub](https://github.com/crandreea))

---

## Running the Application

### 1. Clone the repository
```bash
git clone https://github.com/Mogalina/lung-cancer-detection.git
cd lung-cancer-detection
```

### 2. Run AI model server
```bash
cd lung-cancer-detection/src/ai-model
docker build -t lung-cancer-api .                                                         
docker run -p 6000:6000 lung-cancer-api  
```

### 3. Run backend server
```bash
cd lung-cancer-detection/src/backend
./gradlew build
./gradlew run
```

### 4. Run frontend client
```bash
cd lung-cancer-detection/src/client
./gradlew build
./gradlew run
```

## Results

- Accuracy: 92.8%
- AUC: 0.9913
- Sensitivity: 92.81%
- Specificity: 96.40%

## Sustainable Goals

This project supports:
- UN SDG 3 (Good Health and Well-being)
- UN SDG 9 (Industry, Innovation and Infrastructure)
- UN SDG 17 (Partnerships for the Goals)
