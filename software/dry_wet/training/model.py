import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

def create_road_classifier(num_classes, pretrained_path=None):
    """
    Creates an EfficientNet-B3 model with a custom classifier head for road condition classification.

    Args:
        num_classes (int): The number of output classes.
        pretrained_path (str, optional): Path to the pretrained weights file. 
                                          If None, downloads weights from the internet.
                                          Defaults to None.

    Returns:
        torch.nn.Module: The configured model.
    """
    # Create the model without pretrained weights first
    model = EfficientNet.from_name('efficientnet-b3')

    # Load weights if a path is provided
    if pretrained_path:
        try:
            # Load the state dict from the local .pth file
            state_dict = torch.load(pretrained_path)
            model.load_state_dict(state_dict)
            print(f"Successfully loaded pretrained weights from {pretrained_path}")
        except FileNotFoundError:
            print(f"Warning: Pretrained weights file not found at {pretrained_path}. Training from scratch.")
        except Exception as e:
            print(f"Error loading pretrained weights from {pretrained_path}: {e}. Training from scratch.")
    else:
        # Fallback to downloading from the internet if no local path is given
        print("No local pretrained path provided, attempting to download from the internet.")
        model = EfficientNet.from_pretrained('efficientnet-b3')

    # Replace the final fully connected layer for the dry wet image classification
    model._fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model._fc.in_features, num_classes)
    )
    
    return model