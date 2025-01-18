# Corn Diseases Detection Using ResNeXt Model

## Overview
This project employs the ResNeXt deep learning architecture to detect corn diseases with high accuracy. The model leverages grouped convolutions for efficient feature learning, achieving robust performance on a curated dataset of corn images.

## Features
- Advanced **ResNeXt architecture** for grouped convolutions.
- High accuracy in classifying six corn disease categories.
- Data augmentation for robust training and improved generalization.
- Outputs detailed predictions with confidence levels.
- Visualization tools for accuracy, loss, and confusion matrix.

## Dataset
- **Source**: Corn disease images categorized into six classes, including healthy crops.
- **Preprocessing**:
  - Images are resized to 224x224 pixels.
  - Data augmentation includes rotation, scaling, shearing, and flipping.

## Methodology
1. **Data Preparation**:
   - Augmentation using `ImageDataGenerator` for improved robustness.
   - Separate directories for training and testing datasets.

2. **Model Architecture**:
   - **ResNeXt Block**: Grouped convolutions with cardinality for enhanced feature extraction.
   - Added global average pooling and dense layers for classification.
   - Softmax activation for multi-class predictions.

3. **Training**:
   - Optimized with Adam optimizer and categorical cross-entropy loss.
   - Early stopping and learning rate reduction for efficient training.
   - Trained for 30 epochs with real-time validation monitoring.

4. **Evaluation**:
   - Test accuracy: **95.15%**.
   - Confusion matrix and classification report for performance analysis.

## Dependencies
- Python
- TensorFlow/Keras
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

## Results
- **Accuracy**: 95.15%.
- **Evaluation Metrics**: High precision, recall, and F1-scores across all classes.
- Visualization of confusion matrix, accuracy/loss graphs, and predictions.

## Visualizations
- **Accuracy and Loss Graphs**: Shows training and validation performance.
- **Confusion Matrix**: Highlights class-wise predictions.
- **Sample Predictions**: Images with predicted and actual labels alongside confidence scores.

## Future Improvements
- Expand dataset with additional corn disease categories.
- Optimize architecture for faster inference in real-time scenarios.
- Deploy the model in a mobile or web application for field use.

## Contributing
Contributions are welcome! Fork the repository, implement changes, and submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) for details.

## Acknowledgments
- Thanks to dataset contributors for their valuable efforts.
- TensorFlow/Keras for deep learning frameworks.
- Open-source community for tools and support.
