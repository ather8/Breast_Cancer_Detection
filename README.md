------------------------------
## Breast Histopathology Image Classification (IDC Detection)
This repository contains an end-to-end deep learning pipeline for detecting Invasive Ductal Carcinoma (IDC) from histopathology image patches. By leveraging the InceptionResNetV2 architecture and two-stage fine-tuning, the model automates the detection of malignant tissue patterns with high precision.
------------------------------
## 🚀 How to Run

   1. Clone & Setup
   
   git clone https://github.com/ather8/Breast_Cancer_Detection
   cd idc-classification
   pip install tensorflow opencv-python pandas matplotlib seaborn scikit-learn tf2onnx onnx
   
   2. Dataset
   Download the [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images) and ensure the base_path in the code points to your local directory.
   3. Train & Export
   The script handles data cleaning (removing patients with <100 patches), training, and conversion.
   
   python -m tf2onnx.convert --saved-model saved_model_temp --output model.onnx --opset 13
   
   
------------------------------
## 🏗️ Model & Training Strategy
We utilize Transfer Learning with images upsampled to 75x75 to maximize the feature extraction capabilities of InceptionResNetV2.

* Phase 1: Feature Extraction: Base layers are frozen. We train a custom head (256 units + Dropout) using the Adam optimizer ($LR = 10^{-4}$).
* Phase 2: Fine-Tuning: Unfreezes the base model from layer 600+. Batch Normalization layers remain frozen to maintain statistical stability ($LR = 10^{-5}$).

------------------------------
## 📊 Summary of Results
The model consistently achieves high scores, particularly in Recall, which is critical for medical screening to minimize false negatives.

| Metric | Phase 1 (Head Only) | Phase 2 (Fine-Tuning) |
|---|---|---|
| Accuracy | ~82–85% | ~90–94% |
| AUC Score | ~0.89 | ~0.96–0.99 |
| IDC Recall | ~75% | ~88–98% |
| F1-Score | ~0.80 | ~0.91–0.98 |

------------------------------
## 📦 Technical Stack

* Frameworks: TensorFlow 2.x, Keras
* Image Processing: OpenCV (BGR/RGB conversion), ImageDataGenerator
* Deployment: Exported as model.onnx for production-ready inference.
* Analysis: Pandas, Scikit-learn (Confusion Matrix, Classification Report)

------------------------------
## Future Improvements
To further enhance the model's clinical utility, the following steps are proposed:

* Stain Normalization: Implement algorithms (like Macenko normalization) to reduce variability in staining across different labs.
* Explainable AI (XAI): Integrate Grad-CAM visualizations to highlight the specific regions of the patch that influenced the model's "Cancerous" prediction.
* Multi-scale Analysis: Incorporate patches at different magnifications (e.g., 40x, 100x) to provide broader context to the classifier.
* Attention Mechanisms: Add a transformer-based attention layer to better model the spatial relationships between cells. [1, 2, 3, 4, 5, 6] 

## 📜 License
This project is licensed under the MIT License. See the LICENSE file for details.
------------------------------
