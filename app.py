from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import base64
from io import BytesIO
import bcrypt
from datetime import datetime
import json

app = Flask(__name__)
app.secret_key = '9881560237'

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Flask-Login setup
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    results = db.relationship('Result', backref='user', lazy=True)

# Result model
class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)
    probabilities = db.Column(db.Text, nullable=False)  # Store as JSON string

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Acne classification model
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

# --- ROUTES ---

@app.route('/')
def landing():
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password'].encode('utf-8')
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.checkpw(password, user.password.encode('utf-8')):
            login_user(user)
            return redirect(url_for('index'))
        return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        if password != confirm_password:
            return render_template('register.html', error='Passwords do not match')
        if len(password) < 8:
            return render_template('register.html', error='Password must be at least 8 characters')
        if User.query.filter_by(email=email).first():
            return render_template('register.html', error='Email already registered')
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        new_user = User(name=name, email=email, password=hashed_password.decode('utf-8'))
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/index')
@login_required
def index():
    return render_template('index.html', username=current_user.name)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    data = request.get_json()
    image_data = data['image']
    # Decode image data
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
        probabilities = torch.softmax(output, dim=1)[0].tolist()
    regions = ['forehead', 'cheeks', 'nose', 'chin']
    result = dict(zip(regions, probabilities))
    # Store results in session for suggestions page
    session['detected_regions'] = result
    # Store result in database for user history
    db_result = Result(user_id=current_user.id, probabilities=json.dumps(result))
    db.session.add(db_result)
    db.session.commit()
    return jsonify(result)

@app.route('/suggestions')
@login_required
def suggestions():
    detected_regions = session.get('detected_regions', {})
    return render_template('suggestion.html', detected_regions=detected_regions)

@app.route('/personalized')
@login_required
def personalized():
    results = Result.query.filter_by(user_id=current_user.id).order_by(Result.date.desc()).all()
    # Parse probabilities from JSON string
    parsed_results = []
    for result in results:
        parsed_results.append({
            'date': result.date,
            'probabilities': json.loads(result.probabilities)
        })
    return render_template('personalized.html', results=parsed_results)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
