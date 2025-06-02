# Deep Learning Project Report  
## Project Title: Face Recognition using Deep Learning

### Introduction

Face recognition is a challenging computer vision task that involves identifying or verifying individuals from digital images or videos. Its applications range from security systems to personalized user experiences. This report documents the development of a deep learning model for face recognition, detailing the problem analysis, network building, training, testing, and results.

---

### Problem Analysis and Background Research

Face recognition systems rely on distinguishing unique features from different faces while being robust to variations such as lighting, pose, and occlusions. Deep learning methods, especially convolutional neural networks (CNNs), have achieved state-of-the-art performance in this domain.

**Research Objectives:**
1. Develop a robust face recognition model capable of differentiating between multiple individuals.
2. Address challenges such as overfitting, class imbalance, and variations in facial features.

---

### Building Deep Learning Network

#### Dataset

**Description**  
The dataset consists of three folders (‘train’, ‘test’, and ‘validation’), each containing images categorized into the following classes:
1. Ela
2. Scarlet Johanson
3. Hugh Jackman

**Preprocessing**
1. Face Detection: Haar Cascade classifier was used to detect faces, and cropped images were saved.
2. Normalization: Pixel values were scaled to the range [0, 1].
3. Data Augmentation: Augmentations included random cropping, rotation, horizontal flipping, Gaussian Blur, brightness adjustment, and hue saturation.

---

### Deep Learning Network

#### Network Architecture
- **Input Layer**: 224x224x3 (image size and RGB channels)
- **Convolutional Layers**: Extract spatial features using filters.
- **Pooling Layers**: Reduce spatial dimensions and prevent overfitting.
- **Fully Connected Layers**: Map extracted features to class probabilities.
- **Output Layer**: SoftMax activation for multi-class classification.

#### Loss Function
Cross-entropy loss was selected due to its effectiveness in multi-class classification tasks.

#### Optimizer
Adam optimizer was used for its adaptive learning rate capabilities and efficiency in training deep networks.

#### Hyperparameters
- Learning rate: 0.00001
- Batch size: 3
- Epochs: 50

---

### Training and Evaluation

**Training**
- Split: 70% training, 15% validation, 15% testing.
- Tools: TensorFlow and Keras frameworks.

**Evaluation Metrics**  
Accuracy, precision, recall, and F1-score were chosen as performance metrics.  
The confusion matrix provides insights into class-specific performance, identifying patterns of misclassification critical for face recognition applications.

- Accuracy Curves and Loss Curves

  ![image](https://github.com/user-attachments/assets/2d02e09b-25d0-47d7-9224-527113394eac)

- Precision, Recall, F1-score

  ![image](https://github.com/user-attachments/assets/d8452afa-7631-497d-b130-6bfab4a21f4b)


This classification performance report shows strong results across three categories: Ela, Hugh Jackman, and Scarlet Johanson. All three categories demonstrate high precision (0.90-0.97) and recall (0.90-0.98), resulting in excellent f1-scores around 0.94. The support values indicate varying sample sizes, with Scarlet Johanson having the largest support at 618 samples. The overall model achieves 0.94 accuracy, with consistent macro and weighted averages.

**Confusion Matrix**

![image](https://github.com/user-attachments/assets/d24714c0-7fd9-4de5-a347-3824cd464484)

The confusion matrix highlights the model's performance in classifying "Ela," "Hugh Jackman," and "Scarlet Johanson." It shows strong accuracy for all classes, with 461, 323, and 586 correct classifications, respectively.  Misclassifications occurred, with "Ela" being confused for others 52 times, "Hugh Jackman" 7 times, and "Scarlet Johanson" 32 times. Precision and recall metrics reflect high reliability, such as a 96.6% precision for "Ela" and a 92.2% recall for "Hugh Jackman." Overall, the model demonstrates strong but slightly imbalanced predictions.

---

### Testing

The model was tested on unseen data (test set). The results were as follows:
- Accuracy: 93.77%
- Confusion Matrix:
  - Camera Roll: High accuracy
  - Misclassifications: Ela occasionally misidentified as Hugh Jackman.

---

### Discussions and Conclusions

**Key Observations**
1. The model performed well on the dataset but struggled with similar facial features.
2. Augmentation significantly improved robustness to variations.
3. Fine-tuning the hyperparameters further improved the model’s performance.

**Future Work**
1. Introduce larger datasets for more diversity.
2. Experiment with different loss functions (e.g., triplet loss).
3. Implement real-time face recognition on video streams.

---

### Bias and Ethical Challenges

Bias in the dataset poses a significant ethical concern, as imbalanced representation of different demographics (e.g., ethnicity, gender, or age) can lead to unfair performance disparities.

Additionally, the use of celebrity images may limit applicability to real-world scenarios. Ethical considerations include privacy issues and potential misuse for surveillance.

---

### References

1. Zhang, K., Zhang, Z., Li, Z., & Qiao, Y. (2016). Joint Face Detection and Alignment Using Multi-task Cascaded Convolutional Networks. *IEEE Signal Processing Letters*.
2. MTCNN GitHub Repository: https://github.com/ipazc/mtcnn
3. OpenCV Documentation: https://docs.opencv.org

## Dataset (Not Included)

The dataset used for this project (`DSET/`) is excluded from the repository due to size.

### Expected Folder Structure:
![image](https://github.com/user-attachments/assets/a6dde1ec-3ea2-438b-92ff-aaf6fbe845da)




