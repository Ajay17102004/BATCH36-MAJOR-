import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# ----------------------------
# 1. Generator class (same as training)
# ----------------------------
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# ----------------------------
# 2. Load trained generator
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
generator.load_state_dict(torch.load("generator.pth", map_location=device))
generator.eval()

# ----------------------------
# 3. Image preprocessing
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ----------------------------
# 4. Prediction function
# ----------------------------
def predict_image(image_path, save_path="predicted_mask.png", show=True):
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        mask = generator(input_tensor)

    mask = mask.squeeze().cpu()
    mask_img = transforms.ToPILImage()(mask)
    mask_img.save(save_path)
    print(f"Prediction saved as {save_path}")

    if show:
        plt.subplot(1,2,1)
        plt.imshow(img)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1,2,2)
        plt.imshow(mask_img, cmap="gray")
        plt.title("Predicted Mask")
        plt.axis("off")
        plt.show()

# ----------------------------
# 5. Batch prediction function
# ----------------------------
def predict_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".tif")):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"mask_{filename}")
            predict_image(input_path, save_path=output_path, show=False)
            print(f"Processed {filename}")

# ----------------------------
# 6. Run prediction
# ----------------------------
if __name__ == "__main__":
  
    predict_image("ts1.jpg")  

