"""
Trains a PyTorch image classification model using device-agnostic code.
"""
import os
import torch
import argparse
from torchvision import transforms
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import torch.optim as optim




import data_setup, engine, model_builder, utils


def parse_config():
    parser = argparse.ArgumentParser(description='Train a PyTorch image classification model.')
    
    # Model hyperparameters
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (default: 32)')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs for training (default: 5)')
    parser.add_argument('--hidden_units', type=int, default=10, help='Number of hidden units in the model (default: 10)')
    
    # Dataset and Dataloader parameters
    parser.add_argument('--train_dir', type=str, required=True, help='Directory for training data')
    parser.add_argument('--test_dir', type=str, required=True, help='Directory for testing data')
    parser.add_argument('--workers', type=int, default=2, help='Number of workers for dataloading (default: 2)')
    
    args = parser.parse_args()
    return args


def main():
    # Parse command-line arguments
    args = parse_config()
    
    # Setup device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data augmentations and transformations for the training set
    train_data_transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize first
        transforms.RandomAffine(degrees=15, scale=(0.9, 1.1), translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(p=0.5),  # Apply augmentations after resizing
        transforms.RandomRotation(degrees=15), # Randomly rotate the image by 15 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Random color jitter
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize to [-1, 1]
    ])
    
    # Data augmentations and transformations for the test set
    test_data_transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize all images to 64x64
        transforms.ToTensor()  # Convert image to tensor format
    ])
    
    # Create DataLoader's and get class names
    train_dataloader, _, class_names = data_setup.create_dataloaders(
        train_dir=args.train_dir, 
        test_dir=args.test_dir,
        transform=train_data_transform,
        batch_size=args.batch_size,
        num_workers=args.workers
    )

    # Create DataLoader's and get class names
    _, test_dataloader, _ = data_setup.create_dataloaders(
        train_dir=args.train_dir, 
        test_dir=args.test_dir,
        transform=test_data_transform,
        batch_size=args.batch_size,
        num_workers=args.workers
    )
    
    # Create model
    model = model_builder.TinyVGG(
        input_shape=3,
        hidden_units=args.hidden_units,
        output_shape=len(class_names)
    ).to(device)
    
    # Setup loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    #optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    
    # Start the timer
    start_time = timer()
    
    # Start training with help from engine.py
    model_result = engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=args.num_epochs,
        device=device
    )
    
    # End the timer and print out how long it took
    end_time = timer()
    print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")
    
    # Save the model to file
    utils.save_model(
        model=model,
        target_dir="models",
        model_name="05_going_modular_script_mode_tinyvgg_model.pth"
    )

    utils.plot_loss_curves(model_result)

if __name__ == "__main__":
    main()
