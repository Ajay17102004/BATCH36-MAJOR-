import os
from flask import Flask, request, render_template, send_file
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
generator.load_state_dict(torch.load("generator.pth", map_location=device))
generator.eval()


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


def predict_image(image_path, save_path):
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        mask = generator(input_tensor)

    mask = mask.squeeze().cpu()
    mask_img = transforms.ToPILImage()(mask)
    mask_img.save(save_path)
    return save_path


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400

        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400

        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        output_path = os.path.join(OUTPUT_FOLDER, f"mask_{file.filename}")
        file.save(input_path)

        predict_image(input_path, output_path)

        return send_file(output_path, mimetype="image/png")

    return """
    <!doctype html>
    <title>Retinal Vessel Segmentation</title>
    <h1>Upload Retinal Image</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    """


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
