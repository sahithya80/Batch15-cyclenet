import os
import torch
from torchvision.transforms import ToTensor, Normalize, Compose
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Define the generator architecture
class CycleGenerator(torch.nn.Module):
    def __init__(self, conv_dim=64, n_res_blocks=6):
        super(CycleGenerator, self).__init__()
        # Encoder: Downsampling
        self.layer_1 = self.conv(3, conv_dim, 4)
        self.layer_2 = self.conv(conv_dim, conv_dim * 2, 4)
        self.layer_3 = self.conv(conv_dim * 2, conv_dim * 4, 4)

        # ResNet Blocks
        res_layers = []
        for _ in range(n_res_blocks):
            res_layers.append(self.residual_block(conv_dim * 4))
        self.res_blocks = torch.nn.ModuleList(res_layers)

        # Decoder: Upsampling
        self.layer_4 = self.deconv(conv_dim * 4, conv_dim * 2, 4)
        self.layer_5 = self.deconv(conv_dim * 2, conv_dim, 4)
        self.layer_6 = self.deconv(conv_dim, 3, 4, batch_norm=False)

    def conv(self, in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
        layers = [torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
        if batch_norm:
            layers.append(torch.nn.BatchNorm2d(out_channels))
        return torch.nn.Sequential(*layers)

    def deconv(self, in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
        layers = [torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
        if batch_norm:
            layers.append(torch.nn.BatchNorm2d(out_channels))
        return torch.nn.Sequential(*layers)

    def residual_block(self, conv_dim):
        return torch.nn.ModuleDict({
            'layer_1': self.conv(conv_dim, conv_dim, 3, 1, 1),
            'layer_2': self.conv(conv_dim, conv_dim, 3, 1, 1),
        })

    def forward(self, x):
        out = torch.nn.functional.relu(self.layer_1(x))
        out = torch.nn.functional.relu(self.layer_2(out))
        out = torch.nn.functional.relu(self.layer_3(out))
        for res_block in self.res_blocks:
            out = out + res_block['layer_2'](torch.nn.functional.relu(res_block['layer_1'](out)))
        out = torch.nn.functional.relu(self.layer_4(out))
        out = torch.nn.functional.relu(self.layer_5(out))
        out = torch.tanh(self.layer_6(out))
        return out

# Function to preprocess input images
def preprocess_image(image_path):
    transform = Compose([
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    image = Image.open(image_path).convert("RGB")  # Open the image
    return transform(image).unsqueeze(0)  # Add batch dimension

# Function to postprocess generated images
def postprocess_image(tensor):
    tensor = tensor.squeeze().cpu().numpy()
    tensor = (tensor + 1) / 2  # Rescale from [-1, 1] to [0, 1]
    return np.transpose(tensor, (1, 2, 0))  # Change shape to (H, W, C)

# Main function to run inference
def main():
    # Directory containing the saved models
    save_dir = "saved_models"

    # Initialize the models
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    G_YtoX = CycleGenerator().to(device)

    # Load the saved weights
    G_YtoX.load_state_dict(torch.load(os.path.join(save_dir, "G_YtoX.pth"), map_location=device), strict=False)

    # Set the model to evaluation mode
    G_YtoX.eval()

    # Path to the input CT image
    ct_image_path = r"C:\Users\nagababuj\Desktop\ct_test_google.jpg"  # Replace with your CT image path

    # Preprocess the CT image
    ct_image = preprocess_image(ct_image_path).to(device)

    # Generate an MRI image
    with torch.no_grad():
        mri_image = G_YtoX(ct_image)

    # Postprocess and display the generated MRI image
    generated_mri = postprocess_image(mri_image)
    plt.imshow(generated_mri)
    plt.axis("off")
    plt.title("Generated MRI Image")
    plt.show()

if __name__ == "__main__":
    main()