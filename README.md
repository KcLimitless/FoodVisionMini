# Food Vision Mini 🍕🍔🥗
**Food Vision Mini** is a deep learning project designed to classify images of food(i.e.: ) into three categories: pizza, steak and sushi, using transfer learning with pre-trained models. This notebook provides an end-to-end solution for building a food classification model using PyTorch and torchvision models. It's a simplified version of a larger-scale project, demonstrating the essential techniques required for creating and training image classifiers on custom datasets.

## Project Overview
The Food Vision Mini project involves:
- Loading dataset of food images(subsets of food101 pizza, steak, sushi images).
- Create Datasets and DataLoaders
- Getting the pretrained models, freezing the base layers and changing the classifier head
  (Fine-tuning the models on customized dataset).
- Running several different modelling experiments with various levels of data and tracking the experiments using tensorboard.
- Evaluating model performance through accuracy, loss metrics, and visualizing predictions.
  
## Installation
**Requirements**
- Python 3.8 or higher
- PyTorch
- Torchvision
- Matplotlib
- NumPy
- Jupyter Notebook (if running locally)

**Installation Steps**
1. Clone the repository:
   ``` git clone https://github.com/your-username/FoodVisionMini.git cd FoodVisionMini ```
3. Install the required dependencies
4. Run the notebook

## Usage
Once you have set up your environment, the Food Vision Mini notebook can be executed step-by-step to:
1. Load the Data: Import the dataset and preprocess it for training.
2. Build Models: Use transfer learning to initialize and customize the models.
3. Train Models: Fit the models on the training data and validate on the test data.
4. Evaluate the Models: Assess performance using metrics like accuracy and loss, and visualize predictions

## Model Architecture
The model architecture for Food Vision Mini is based on the EfficientNet family of models from the torchvision library. EfficientNet models are known for their balance between high accuracy and computational efficiency, making them well-suited for image classification tasks. In this project, three versions of EfficientNet are utilized:
1. EfficientNet-B0 (torchvision.models.efficientnet_b0)
2. EfficientNet-B2 (torchvision.models.efficientnet_b2)
3. EfficientNet-B3 (torchvision.models.efficientnet_b3)

Each of these models is pretrained on ImageNet, leveraging transfer learning to accelerate training and improve results.

## Contributing 
If you’d like to contribute to this project, feel free to fork the repository and submit a pull request. Issues and feedback are also welcome!

## Acknowledgments
- PyTorch: The project uses PyTorch's deep learning framework for model training and evaluation.
- Pre-trained Models: EfficientNet models are used from Torchvision’s pre-trained model library.
- Inspired by Mr. D Bourke’s PyTorch Deep Learning course.

## File Structure

├── data/                    # Contains dataset files (not included in the repo)

├── models/                  # Directory to save trained models

├── scripts/                 # Custom Python scripts for data loading, model building, etc.

├── Food_Vision_Mini.ipynb    # Main Jupyter notebook for the project

└── README.md                # Project README file

