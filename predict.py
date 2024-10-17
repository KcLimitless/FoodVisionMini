
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
import argparse
from model_builder import TinyVGG  # Ensure TinyVGG is imported

# Parse arguments
def parse_config():
    parser = argparse.ArgumentParser(description='Predict an image using a trained PyTorch model.')
    
    # Add arguments
    parser.add_argument('--model', type=str, default="models/05_going_modular_script_mode_tinyvgg_model.pth", help='Path to the trained model file (.pth)')
    parser.add_argument('--image', type=str, required=True, help='Path to the image to perform inference on')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Device to run inference on (default: auto-detect cuda if available)')

    args = parser.parse_args()
    return args

# Function to load in the model
def load_model(filepath, input_shape, hidden_units, output_shape, device):
    # Initialize the model with given hyperparameters
    model = TinyVGG(input_shape=input_shape, hidden_units=hidden_units, output_shape=output_shape)

    print(f"[INFO] Loading in model from: {filepath}")
    
    # Load the state dictionary from the given file path
    model.load_state_dict(torch.load(filepath, weights_only=True, map_location=device))

    return model

# Prediction and plotting function
def pred_and_plot_image(model: torch.nn.Module, image_path: str, class_names: list, device: str):
    """Make a prediction on a target image with a trained model and plot the image with its prediction."""
    # Transform pipeline (resize image)
    custom_image_transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
    ])
    
    # Load image
    target_image = torchvision.io.read_image(image_path).type(torch.float32)
    target_image = target_image / 255.0  # Normalize image to [0, 1] range

    # Apply transformations
    target_image = custom_image_transform(target_image)
    
    # Move model to the appropriate device
    model.to(device)
    model.eval()

    with torch.inference_mode():
        # Add batch dimension (1x, C, H, W)
        target_image = target_image.unsqueeze(0).to(device)
        
        # Make prediction
        pred_logits = model(target_image)
        pred_probs = torch.softmax(pred_logits, dim=1)
        pred_label = torch.argmax(pred_probs, dim=1)

        pred_label_class = class_names[pred_label]

    print(f"[INFO] Pred class: {pred_label_class}, Pred prob: {pred_probs.max():.3f}")         


    # Plot image with prediction
    #plt.imshow(target_image.squeeze().permute(1, 2, 0).cpu())  # Convert back to HWC and plot
    #plt.title(f"Pred: {class_names[pred_label.cpu()]} | Prob: {pred_probs.max().cpu():.3f}")
    #plt.axis('off')
    #plt.show()

# Main function
def main():
    # Parse command-line arguments
    args = parse_config()
    
    # Define the model parameters
    class_names = ["pizza", "steak", "sushi"]
    input_shape = 3  # Number of input channels (e.g., RGB)
    hidden_units = 64  # Example number of hidden units, adjust as per your model
    output_shape = len(class_names)  # Number of output classes

    # Get the image path
    IMG_PATH = args.image
    print(f"[INFO] Predicting on {IMG_PATH}")

    # Load the model using the load_model function
    model = load_model(filepath=args.model, 
                       input_shape=input_shape, 
                       hidden_units=hidden_units, 
                       output_shape=output_shape, 
                       device=args.device)

    # Perform prediction and plot the result
    pred_and_plot_image(model=model, image_path=args.image, class_names=class_names, device=args.device)

# Run the script
if __name__ == "__main__":
    main()
