import os
import cv2
import torch
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms

class CustomCNNModel(nn.Module):

    def __init__(self, input_dim, num_classes):
        super(CustomCNNModel, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        # PURPOSE: Feature extraction
        self.conv_layers = nn.Sequential(

            # C1
            nn.Conv2d(
                in_channels=3,   # Input has 3 channels (RGB)
                out_channels=32, # Generate 32 feature maps
                kernel_size=3,   # Each filter has a spatial size of 3 × 3
                stride=1,        # Move the filter 1 pixel at a time
                padding=1        # Add 1 pixel of padding around the image to preserve its spatial dimensions
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # C2
            nn.Conv2d(
                in_channels=32,  # Input has 32 channels
                out_channels=64, # Generate 64 feature maps
                kernel_size=3,   # Each filter has a spatial size of 3 × 3
                stride=1,        # Move the filter 1 pixel at a time
                padding=1        # Add 1 pixel of padding around the image to preserve its spatial dimensions
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # C3
            nn.Conv2d(
                in_channels=64,   # Input has 64 channels
                out_channels=128, # Generate 128 feature maps
                kernel_size=3,    # Each filter has a spatial size of 3 × 3
                stride=1,         # Move the filter 1 pixel at a time
                padding=1         # Add 1 pixel of padding around the image to preserve its spatial dimensions
            ),

            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # C4
            nn.Conv2d(
                in_channels=128,   # Input has 128 channels
                out_channels=256,  # Generate 256 feature maps
                kernel_size=3,     # Each filter has a spatial size of 3 × 3
                stride=1,          # Move the filter 1 pixel at a time
                padding=1          # Add 1 pixel of padding around the image to preserve its spatial dimensions
            ),

            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self._to_linear = None
        self._get_conv_output(self.input_dim)

        # PURPOSE: Classification
        self.fc_layers = nn.Sequential(
            nn.Linear(self._to_linear, 512),
            nn.ReLU(),
            # nn.Dropout(0.2)
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes),
        )

    # Calculate the number of features produced by the CVN layers after processing an input image.
    def _get_conv_output(self, input_dim=128):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, input_dim, input_dim)
            output = self.conv_layers(dummy_input)
            self._to_linear = output.view(1, -1).size(1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

class ImageClassifier():
    def __init__(self, model_path, class_names=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CustomCNNModel(input_dim=128, num_classes=3).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Define custom class names or default class names
        if class_names is None:
            self.class_names = {0: 'Dog', 1: 'Cat', 2: 'Person'}
        else:
            self.class_names = class_names

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
            ])

    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            _, predicted = torch.max(output, 1)
            label = self.class_names[predicted.item()]
        img = cv2.imread(image_path)
        cv2.putText(img=img,                           # Original image on which the text will be drawn
                    text=label,                        # Predicted class name
                    org=(10,50),                      # Position of the text (x=10, y=50)
                    fontFace=cv2.FONT_HERSHEY_COMPLEX, # Font style
                    fontScale=1,                       # Font size multiplier
                    color=(0, 0, 255),                 # Text color in BGR format (red)
                    thickness=2)                       # Thickness of the text

        cwd = os.getcwd()
        output_path = os.path.join(cwd, "images", "labelled_image.jpg")
        cv2.imwrite(output_path, img)
        # output_path = os.path.join(cwd, output_path)

        return label, output_path
