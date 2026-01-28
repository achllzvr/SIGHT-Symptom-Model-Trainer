# 👁️ SIGHT: AI Model Training & Integration Guide

**Project:** SIGHT Mobile Health Application  
**Module:** Edge AI (Eye Symptom Detection)  
***Target Device:** Android 16 (Minimum of Android 10)

---

## 📂 Phase 1: Environment & Dataset Setup

### 1. Python Environment
Ensure you have the necessary libraries installed to run the training scripts.
> **Action:** Open your terminal in the root folder (`SIGHT-SYMPTOM-MODEL-TRAINER`) and install the requirements.
>
> *Required Libraries:* `tensorflow`, `numpy`, `pillow`
* **Action:** Run the following command in your terminal:
    ```bash
    pip install tensorflow numpy pillow scipy
    ```
    ```bash
    // For MacOS troubles.
    pip install tensorflow-macos tensorflow-metal numpy pillow
    // Make sure that python version is 3.11 because at the time of writing this, 3.14 does not support tflite.
    ```

### 2. Dataset Preparation
You need to populate the dataset folders with your collected images before training.

* **Directory:** `sight_dataset/`
* **Action:** Place your images into the respective subfolders:
    * 📁 `sight_dataset/cataract/` *(Cataract images)*
    * 📁 `sight_dataset/healthy/` *(Normal eye images)*
    * 📁 `sight_dataset/ptosis/` *(Eyelid drooping images)*
    * 📁 `sight_dataset/uveitis/` *(Redness/Infection images)*

> **⚠️ Note:** Ensure the `eyelid` folder is empty or deleted if "ptosis" covers your eyelid drooping data to avoid confusion.

---

## 🧠 Phase 2: Training the AI Model

### 1. Train the Base Model
This script loads the **MobileNetV2** architecture, feeds in your dataset, and saves the heavy model.

* **File to Run:** `train_sight.py`
* **Action:** Run the following command in your terminal:
    ```bash
    python train_sight.py
    ```
* **Output:** Generates `sight_model_full.h5` in your root directory.

### 2. Quantize the Model (Make it Mobile-Ready)
This script compresses the heavy model into a format readable by Android/iOS devices (**TFLite**).

* **File to Run:** `convert_model.py`
* **Action:** Run the following command in your terminal:
    ```bash
    python convert_model.py
    ```
* **Output:** Generates `sight_model_quant.tflite` in your root directory.

---

## 📱 Phase 3: Flutter Integration

Move the trained model and code files into your actual Flutter application project.

### 1. Asset Setup
* **Action A:** Copy `sight_model_quant.tflite` (from root) ➔ `assets/` (in Flutter app).
* **Action B:** Create a new file `assets/labels.txt`.
    * *Content:* List your 4 classes (**Healthy**, **Uveitis**, **Cataract**, **Ptosis**) in the **exact order** they appeared in the terminal output ("Class Indices") from Phase 2.

### 2. Dependencies
* **Reference File:** `Flutter Code/Code Files/pubspec.yaml`
* **Action:** Copy the dependency lines (e.g., `camera`, `tflite_flutter`, `google_mlkit_face_detection`, `image`) into your project's `pubspec.yaml`.

### 3. Image Conversion Utilities
Handles the complex math of converting raw camera feed (YUV) to AI-readable format (RGB).
* **Reference File:** `Flutter Code/Code Files/image_utils.dart`
* **Action:** Copy to `lib/utils/image_utils.dart`.

### 4. UI Overlay (The Green Box)
Contains the "Safe Area" visual guide and "Left/Right Eye" labels.
* **Reference File:** `Flutter Code/Code Files/camera_overlay.dart`
* **Action:** Copy to `lib/widgets/camera_overlay.dart`.

### 5. The AI Logic Service
**The Brain.** Handles model loading, face detection, eye cropping, and the dual-eye prediction loop.
* **Reference File:** `Flutter Code/Code Files/ai_service.dart`
* **Action:** Copy to `lib/services/ai_service.dart`.
    * *Check:* Ensure import paths inside match your project structure.

### 6. Main Implementation
Stitches everything together in the Camera Screen.
* **Reference File:** `Flutter Code/Code Files/main.dart`
* **Action:** Use as a reference for your Camera Screen. Look specifically at the `_aiService.analyzeFrame(image)` call.

---

## 📊 Phase 4: Verification & Testing

### Accuracy Report Generation (For Capstone Paper)

1.  **Test Data:** Collect **20 new eye images** (5 per disease) that were **not** in the training dataset.
2.  **Manual Check:** Run these images through the app.
3.  **Confusion Matrix:**
    * Compare **Actual Class** (Rows) vs. **Predicted Class** (Columns).
    * *Formula:*
        ```
        Accuracy = (Correct Predictions / Total Images) * 100
        ```