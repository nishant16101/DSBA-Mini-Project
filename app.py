from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import base64
from io import BytesIO

app = Flask(__name__)

# Define the model architecture
class AcneRegionClassifier(nn.Module):
    def __init__(self):
        super(AcneRegionClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 4)  # 4 output classes
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = x.view(-1, 256 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Load the model
model = AcneRegionClassifier()
model.load_state_dict(torch.load('best_acne_model .pt', map_location=torch.device('cpu')))
model.eval()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image']
    
    #Decode image data
    image = Image.open(BytesIO(base64.b64decode(image_data.split(',')[1])))
    
    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0].tolist()  # Convert to list
    
    regions = ['forehead', 'cheeks', 'nose', 'chin']
    result = dict(zip(regions, probabilities))
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True) 
