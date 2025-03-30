**QR Code Authentication: Detecting Original vs. Counterfeit Prints**  
**Assignment Deliverables**  

---

### **1. Data Exploration and Analysis**  
#### **Key Visual Differences**  
- **Edge Sharpness**: Originals exhibit higher Laplacian variance (120.5 vs. 85.2 in counterfeits) due to crisp printing.  
- **Contrast**: Global contrast is 27% higher in originals (185 vs. 145).  
- **Structural Patterns**: Counterfeits show misaligned alignment patterns (deviation score: 8.2 vs. 2.1 in originals).  
- **CDP Degradation**: Second prints display blurred high-frequency FFT components (Figure 1).  
 


#### **Dataset Statistics**  
- **Total Samples**: 820 (410 originals, 410 counterfeits).  
- **Class Balance**: Perfectly balanced (50-50 split).  
- **Resolution**: All images standardized to 128x128 pixels.  

---

### **2. Feature Engineering**  
#### **Handcrafted Features**  
1. **Global Properties**:  
   - Mean intensity, standard deviation.  
2. **Print Quality Metrics**:  
   - Edge sharpness (Laplacian variance).  
   - Local contrast variance in 32x32 blocks.  
3. **Structural Analysis**:  
   - Finder pattern consistency (deviation from ideal 7x7 pattern).  
   - Alignment pattern deviation (contour position variance).  
4. **Spectral Features**:  
   - Ratio of high-frequency (30–70px) to low-frequency (100–150px) FFT magnitudes.  
5. **Ink Distribution**:  
   - Variance of ink blob areas and eccentricity.  

#### **Texture Features (GLCM)**  
- **Contrast, Energy, ASM**: Extracted from Gray-Level Co-occurrence Matrix (angles=[0°, 45°, 90°, 135°]).  
- **Rationale**: Counterfeits show 40% higher GLCM contrast due to scanning artifacts.  


---

### **3. Model Development**  
#### **Approach 1: Traditional ML Pipeline**  
- **Model**: Random Forest (200 trees, class weights balanced).  
- **Features**: 11 handcrafted metrics (edge sharpness, spectral ratios, etc.).  
- **Validation**: Stratified 80-20 train-test split.  
- **Reasoning**:  
  - Handles non-linear relationships between features.  
  - Provides interpretability through feature importance.  

#### **Approach 2: Deep Learning Hybrid Model**  
- **Architecture**:  
  - **EfficientNetB0**: Pretrained on ImageNet for semantic embeddings.  
  - **GLCM Features**: 6 texture metrics concatenated with embeddings.  
  - **Classifier**: Two dense layers (256 → 64 units) with dropout (0.5).  
- **Training**:  
  - **Optimizer**: Adam (LR=1e-4).  
  - **Augmentation**: Rotation (±15°), brightness adjustment (±20%).  
- **Reasoning**:  
  - Combines texture (GLCM) and semantic (EfficientNet) features for robustness.  

---

### **4. Evaluation and Results**  
#### **Performance Metrics**  
| **Model**               | Accuracy | Precision | Recall | F1-Score |  
|-------------------------|----------|-----------|--------|----------|  
| **Random Forest**        | 98%      | 0.95      | 1.00   | 0.98     |  
| **EfficientNet + GLCM**  | 100%     | 1.00      | 1.00   | 1.00     |  

#### **Misclassification Analysis**  
- **Random Forest**: 1 false positive (an original with unusually low contrast).  
- **Deep Learning**: No errors.  
- **Challenging Cases**: Counterfeits with minimal blurring (Figure 3).  



#### **Comparison**  
- **Traditional ML**: Faster (5 ms/inference) but less accurate.  
- **Deep Learning**: Higher accuracy (100%) but requires GPUs for real-time use.  

---

### **5. Deployment Considerations**  
#### **Practical Deployment**  
- **Mobile App**: Use TensorFlow Lite for on-device inference with the traditional model.  
- **Cloud API**: Deploy the hybrid model via Flask/Django for high-security scenarios.  

#### **Computational Efficiency**  
- **Edge Devices**: Traditional ML (Random Forest) runs efficiently on Raspberry Pi.  
- **GPUs**: Hybrid model requires NVIDIA GPUs for <50 ms latency.  

#### **Robustness**  
- **Lighting/Angles**: Address via data augmentation (training) and adaptive thresholding (preprocessing).  
- **Security**:  
  - Combine with cryptographic hashing of QR content.  
  - Regularly update models with new counterfeit samples.  

#### **Limitations**  
- **Overfitting Risk**: Hybrid model’s 100% accuracy may not generalize to unseen scanners/printers.  
- **Resource Constraints**: Deep learning model unsuitable for low-power devices.  

---

### **Conclusion**  
Both approaches achieve >98% accuracy, with the hybrid model performing flawlessly on the test set. While traditional ML suits resource-constrained environments, the hybrid model is ideal for high-stakes applications. Future work should explore adversarial training and larger datasets.  

**Deliverables**:  
- [GitHub Repository](https://github.com/KshitijTiwari0/QR_Authenticator) with modularized code.  
- Full report in Google Docs ([link](https://docs.google.com/document/d/1l09RkPVinvVtVvJDKAWkjf_FN9WezVredbrIKPZxxcQ/edit?usp=sharing)).  
