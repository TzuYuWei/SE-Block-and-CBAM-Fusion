# CycleGAN-Deraining-SE-CBAM
## ATTENTION-ENHANCED CYCLEGAN FOR IMAGE DERAINING

**Python | PyTorch | GAN | Computer Vision | Image Restoration**

---

# PROJECT INTRODUCTION

This project focuses on improving **IMAGE DERAINING PERFORMANCE** by enhancing the **CYCLEGAN** architecture with an **ATTENTION MECHANISM**.

The model aims to better capture **RAIN STREAK FEATURES** and preserve **BACKGROUND DETAILS** during the deraining process.

By integrating **ATTENTION MODULES** into the **GENERATOR NETWORK**, the model can focus on important image regions, leading to improved **VISUAL QUALITY** and more effective **RAIN REMOVAL**.

---

# MODEL ARCHITECTURE

The project is based on the **CYCLEGAN FRAMEWORK** and incorporates several key components commonly used in **DEEP LEARNING IMAGE TRANSLATION MODELS**.

### ATTENTION MECHANISM
Integrates **SE BLOCK** and **CBAM (CONVOLUTIONAL BLOCK ATTENTION MODULE)** to jointly model **CHANNEL ATTENTION** and **SPATIAL ATTENTION**, enabling the network to better emphasize **RAIN-AFFECTED REGIONS** and important **IMAGE FEATURES**.

### RESNET BLOCKS
Used in the **GENERATOR NETWORK** to improve **FEATURE EXTRACTION**, **GRADIENT FLOW**, and **MODEL STABILITY**.

### PATCHGAN DISCRIMINATOR
Evaluates **LOCAL IMAGE PATCHES** instead of the entire image to better capture **HIGH-FREQUENCY TEXTURES** and improve **IMAGE REALISM**.

### IMAGE POOL
Stores previously generated images to **STABILIZE ADVERSARIAL TRAINING** and reduce **MODEL OSCILLATION** during GAN training.

### LOSS FUNCTIONS
Utilizes **MULTIPLE LOSS FUNCTIONS** to improve the quality of generated images.  
The training objective goes beyond the standard **CYCLE CONSISTENCY LOSS**, incorporating additional loss components to further enhance **IMAGE GENERATION QUALITY** and **TRAINING STABILITY**.

---

# PROJECT STRUCTURE

---

# TRAINING PIPELINE

1. **Input Rainy Images**
2. **Generator (ResNet + Attention Modules)**
3. **CycleGAN Translation Process**
4. **PatchGAN Discriminator Evaluation**
5. **Loss Calculation (Multiple Loss Functions)**
6. **Model Optimization**

---

# RESULTS

Example deraining results:

Rainy Image → Derained Image

### Image Deraining Example

| Rainy Image | Derained Image | Ground Truth |
|-------------|---------------|--------------|
| ![](Images/result1_rain.png) | ![](Images/result1_clean.png) | ![](Images/result1_groundtruth.png) |
| ![](Images/result2_rain.png) | ![](Images/result2_clean.png) | ![](Images/result2_groundtruth.png) |

---

# RESEARCH BACKGROUND

This project was developed during my graduate studies focusing on **GENERATIVE AI** and **COMPUTER VISION**.

The research related to this work was accepted by the journal:

**Journal of Internet Technology**

Paper Title:

**Enhancing CycleGAN for Image Deraining through SE Block and CBAM Fusion**

---

# FUTURE IMPROVEMENTS

- Improve **RAIN STREAK DETECTION**
- Explore **TRANSFORMER-BASED ATTENTION**
- Enhance **HIGH-RESOLUTION IMAGE GENERATION**

---

# AUTHOR

Wei Tzu-Yu  
Master of Information Network Engineering  
Longhua University of Science and Technology