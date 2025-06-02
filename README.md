# -Face-Recognition-using-Deep-Learning-
Problem Analysis and Background Research 
Face recognition systems rely on distinguishing unique features from different faces while being robust to 
variations such as lighting, pose, and occlusions. Deep learning methods, especially convolutional 
neural networks (CNNs), have achieved state-of-the-art performance in this domain. 
Research Objectives: 
1. Develop a robust face recognition model capable of differentiating between multiple individuals. 
2. Address challenges such as overfitting, class imbalance, and variations in facial features. 
Building Deep Learning Network 
Dataset 
Description 
The dataset consists of three folders (‘train’, ‘test’, and ‘validation’), each containing images categorized 
into the following classes: 
1. Ela 
2. Scarlet Johanson 
3. Hugh Jackman 
Preprocessing 
1. Face Detection: Haar Cascade classifier was used to detect faces, and cropped images were 
saved. 
2. Normalization: Pixel values were scaled to the range [0, 1]. 
3. Data Augmentation: Augmentations included random cropping, rotation, horizontal flipping, 
Gaussian Blur, brightness adjustment, and hue saturation.

Deep Learning Network 
Network Architecture 
• Input Layer: 224x224x3 (image size and RGB channels) 
• Convolutional Layers: Extract spatial features using filters. 
• Pooling Layers: Reduce spatial dimensions and prevent overfitting. 
• Fully Connected Layers: Map extracted features to class probabilities. 
• Output Layer: SoftMax activation for multi-class classification. 
Loss Function 
Cross-entropy loss was selected due to its effectiveness in multi-class classification tasks. 
Optimizer 
Adam optimizer was used for its adaptive learning rate capabilities and efficiency in training deep 
networks. 
Hyperparameters 
• Learning rate: 0.00001 
• Batch size: 3 
• Epochs: 50 
Training and Evaluation 
Training 
• Split: 70% training, 15% validation, 15% testing. 
• Tools: TensorFlow and Keras frameworks. 
Evaluation Metrics 
For face recognition, accuracy, precision, recall, and F1-score were chosen as performance metrics. 
Accuracy measures the overall correctness, while precision and recall assess the model's ability to 
identify faces correctly without false positives or negatives. The F1-score, as a harmonic mean of 
precision and recall, balances these metrics, making it suitable for imbalanced datasets. Additionally, 
the confusion matrix provides insights into class-specific performance, identifying patterns of 
misclassification critical for face recognition applications, such as security or identity verification.
• Accuracy Curves and Loss Curves 
![image](https://github.com/user-attachments/assets/534b051a-3db0-46ce-ae4a-94557196b91c)
• Precision, Recall, F1-score 
![image](https://github.com/user-attachments/assets/3133693b-334c-43dd-9370-5de55361ef6d)

