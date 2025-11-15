from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from werkzeug.utils import secure_filename 
import os
import base64
import json
import requests
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv
from psycopg2 import pool
from contextlib import contextmanager
import io
from PIL import Image
load_dotenv()  # Load environment variables

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your_secret_key')

# Database connection pool setup
db_pool = pool.SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    host=os.getenv('DB_HOST', 'eep-orange-flower-a85zc9oh-pooler.eastus2.azure.neon.tech'),
    database=os.getenv('DB_NAME', 'neondb'),
    user=os.getenv('DB_USER', 'neondb_owner'),
    password=os.getenv('DB_PASSWORD', 'npg_wrUq0Q3jLEZk')
)

@contextmanager
def get_db_connection():
    conn = db_pool.getconn()
    try:
        yield conn
    finally:
        db_pool.putconn(conn)

@contextmanager
def get_db_cursor(commit=False):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            yield cursor
            if commit:
                conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

# Improved model loading with error handling
def load_or_train_model():
    model_path = 'models/crop_model.pkl'
    data_path = 'models/crop_data.csv'
    
    try:
        # First try loading with joblib (better for scikit-learn models)
        model = joblib.load(model_path)
        print("Model loaded successfully with joblib")
        return model
    except Exception as e:
        print(f"Joblib load failed: {e}. Trying pickle...")
        try:
            # Fallback to pickle with numpy compatibility workaround
            import sys
            if 'numpy._core' not in sys.modules:
                sys.modules['numpy._core'] = np.core
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print("Model loaded successfully with pickle")
            return model
        except Exception as e:
            print(f"Pickle load failed: {e}. Training new model...")
            try:
                df = pd.read_csv(data_path)
                X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
                y = df['label']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                
                # Save with both joblib and pickle for future compatibility
                joblib.dump(model, model_path)
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                
                print("New model trained and saved")
                return model
            except Exception as e:
                print(f"Model training failed: {e}")
                raise RuntimeError("Failed to load or train model")

# Load the model when starting the app
crop_model = load_or_train_model()
        
@app.route('/')
def landing():
    return render_template('index.html')



def get_weather_data(location):
    WEATHERAPI_KEY = os.getenv('WEATHERAPI_KEY', '6f165948e0e04eeba1214014252204')
    try:
        # Construct API URL based on input type (city name or coordinates)
        if isinstance(location, dict) and 'lat' in location:
            url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHERAPI_KEY}&q={location['lat']},{location['lon']}&days=7&aqi=no&alerts=no"
        else:
            url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHERAPI_KEY}&q={location}&days=7&aqi=no&alerts=no"
        
        response = requests.get(url)
        data = response.json()
        
        if 'error' in data:
            return {'error': data['error']['message']}
        
        # Process the data
        processed_data = {
            'location': f"{data['location']['name']}, {data['location']['country']}",
            'current': {
                'temp_c': data['current']['temp_c'],
                'feelslike_c': data['current']['feelslike_c'],
                'condition': data['current']['condition']['text'],
                'icon': data['current']['condition']['icon'],
                'humidity': data['current']['humidity'],
                'rainfall': data['current']['precip_mm'],
                'wind_kph': data['current']['wind_kph'],
            },
            'hourly': [],
            'daily': []
        }
        
        # Process hourly data (next 24 hours)
        for hour in data['forecast']['forecastday'][0]['hour']:
            time = datetime.strptime(hour['time'], '%Y-%m-%d %H:%M').strftime('%H:%M')
            processed_data['hourly'].append({
                'time': time,
                'temp_c': hour['temp_c'],
                'humidity': hour['humidity'],
                'rainfall': hour['precip_mm'],
                'condition': hour['condition']['text'],
                'icon': hour['condition']['icon'],
                'chance_of_rain': hour['chance_of_rain']
            })
        
        # Process daily forecast
        for day in data['forecast']['forecastday']:
            date = datetime.strptime(day['date'], '%Y-%m-%d').strftime('%m/%d')
            weekday = datetime.strptime(day['date'], '%Y-%m-%d').strftime('%a')
            processed_data['daily'].append({
                'date': date,
                'day': weekday,
                'high': day['day']['maxtemp_c'],
                'low': day['day']['mintemp_c'],
                'humidity': day['day']['avghumidity'],
                'total_rainfall': day['day']['totalprecip_mm'],
                'condition': day['day']['condition']['text'],
                'icon': day['day']['condition']['icon'],
                'chance_of_rain': day['day']['daily_chance_of_rain']
            })
        
        return processed_data
        
    except Exception as e:
        return {'error': str(e)}
@app.route('/get_weather', methods=['POST'])
def get_weather():
    data = request.get_json()
    if 'city' in data:
        weather_data = get_weather_data(data['city'])
    elif 'lat' in data and 'lon' in data:
        weather_data = get_weather_data({'lat': data['lat'], 'lon': data['lon']})
    else:
        weather_data = {'error': 'Invalid request parameters'}
    return jsonify(weather_data)

@app.route('/weather')
def weather():
    if 'user_id' in session:
        return render_template('weather.html')

@app.route('/home')
def home():
    if 'user_id' in session:
        return render_template('home.html')
    return render_template('login.html')

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to Check Allowed File Extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST': 
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        
        # Handle File Upload
        if 'profile_pic' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['profile_pic']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Store user details and profile picture filename in MySQL
            with get_db_cursor(commit=True) as cursor:
                cursor.execute("INSERT INTO users (name, email, password, profile_pic) VALUES (%s, %s, %s, %s)", 
                               (name, email, password, filename))

            flash('Registration successful! Please log in.')
            return redirect(url_for('login'))
        
        flash('Invalid file type. Allowed types: png, jpg, jpeg, gif')
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        with get_db_cursor() as cursor:
            cursor.execute("SELECT id, name, profile_pic FROM users WHERE email = %s AND password = %s", (email, password))
            user = cursor.fetchone()

        if user:
            session['user_id'] = user[0]
            session['username'] = user[1]
            session['profile_pic'] = user[2] if user[2] else 'default_profile.png'  # Store in session
            return redirect(url_for('weather'))
        else:
            flash("Invalid email or password")

    return render_template('login.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user_id' in session:
        if request.method == 'POST':
            # Get form data
            n = float(request.form['n'])
            p = float(request.form['p'])
            k = float(request.form['k'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])

            # Predict the crop
            features = [[n, p, k, temperature, humidity, ph, rainfall]]
            predicted_crop = crop_model.predict(features)[0]

            # Store prediction in database
            with get_db_cursor(commit=True) as cursor:
                cursor.execute(
                    "INSERT INTO predictions (user_id, nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall, predicted_crop) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (session['user_id'], n, p, k, temperature, humidity, ph, rainfall, predicted_crop)
                )

            # Crop dictionary
            crop_images = {
                'rice': 'crop_images/rice.png',
                'wheat': 'crop_images/wheat.png',
                'maize': 'crop_images/maize.jpeg',
                'apple': 'crop_images/apple.jpg',
                'barley': 'crop_images/Barley.jpg',
                'banana': 'crop_images/banana.jpeg',
                'chickpea': 'crop_images/chickpea.jpg',
                'coconut': 'crop_images/coconut.jpg',
                'coffee': 'crop_images/coffee.jpg',
                'cotton': 'crop_images/cotton.jpeg',
                'grapes': 'crop_images/grapes.jpeg',
                'sugarcane': 'crop_images/Sugarcane.png',
                'chili': 'crop_images/Chili.png',
                'potato': 'crop_images/Potato.png',
                'tomato': 'crop_images/Tomato.png',
                'soybean': 'crop_images/Soybean.png',
                'kidneybeans': 'crop_images/kidneybeans.jpeg',
                'lentil': 'crop_images/lentil.jpeg',
                'mango': 'crop_images/mango.jpeg',
                'mothbeans': 'crop_images/mothbeans.jpeg',
                'mungbeans': 'crop_images/mungbeans.jpeg',
                'muskmelon': 'crop_images/muskmelon.jpeg',
                'orange': 'crop_images/orange.jpeg',
                'papaya': 'crop_images/papaya.jpeg',
                'pigeonpeas': 'crop_images/pigeonpeas.jpeg',
                'pomegranate': 'crop_images/pomergranate.jpg',
                'watermelon': 'crop_images/watermelon.jpg'
            }

            crop_image = crop_images.get(predicted_crop, 'crop_images/default.jpg')

            return render_template('result1.html', 
                                crop=predicted_crop, 
                                crop_image=crop_image,
                                n=n, p=p, k=k,
                                temperature=temperature,
                                humidity=humidity,
                                ph=ph,
                                rainfall=rainfall)
        return render_template('predict.html')
    return redirect(url_for('login'))

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')

@app.route('/contactus', methods=['GET', 'POST'])
def contactus():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        
        # Store user details in the database
        with get_db_cursor(commit=True) as cursor:
            cursor.execute("INSERT INTO contact1 (name, email, message) VALUES (%s, %s, %s)", (name, email, message))
        flash('Thank you for reaching out! We will get back to you soon.')
        return redirect(url_for('contactus'))
    return render_template('contactus.html')
    
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
MODEL_NAME = "google/gemini-2.5-flash-image"

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file:
            # Save uploaded image
            filename = secure_filename(file.filename)
            upload_dir = os.path.join('static', 'uploads')
            os.makedirs(upload_dir, exist_ok=True)
            img_path = os.path.join(upload_dir, filename)
            file.save(img_path)

            try:
                # Convert image to Base64 for OpenRouter
                with open(img_path, "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

                # --- Prompt for AI ---
                prompt = """Analyze this crop image and provide the following in MARKDOWN format:

**Crop Name:** [Identify the crop species]
**Disease:** [Name of the disease or 'No Disease Detected']
**Recommendations:**
- [List of pesticides or fungicides]
- [Preventive measures]

Include appropriate emojis 🌱🌾."""

                # --- Call OpenRouter API ---
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "http://localhost:5000",
                        "X-Title": "Crop Disease Detector"
                    },
                    data=json.dumps({
                        "model": MODEL_NAME,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{encoded_image}"
                                        }
                                    }
                                ]
                            }
                        ]
                    })
                )

                response_json = response.json()

                # --- Handle API errors ---
                if "error" in response_json:
                    error_msg = response_json["error"].get("message", "Unknown API error")
                    return render_template(
                        'error.html',
                        error_message=f"AI Error: {error_msg}",
                        recovery_tip="Try uploading a clearer crop image."
                    )

                # --- Extract AI analysis safely ---
                message_content = response_json['choices'][0]['message'].get('content', '')
                if isinstance(message_content, list):
                    content = ''.join([item.get('text', '') for item in message_content if isinstance(item, dict)])
                else:
                    content = message_content

                # --- Parse AI Response ---
                result_data = {
                    'crop': 'Unknown Crop',
                    'disease': 'No Disease Detected',
                    'recommendations': [],
                    'image': img_path,
                    'status_icon': '⚠️'
                }

                lines = content.split('\n')
                for line in lines:
                    if '**Crop Name:**' in line:
                        result_data['crop'] = line.split('**Crop Name:**')[-1].strip()
                    elif '**Disease:**' in line:
                        result_data['disease'] = line.split('**Disease:**')[-1].strip()
                        result_data['status_icon'] = '✅' if 'No Disease' in result_data['disease'] else '⚠️'
                    elif '**Recommendations:**' in line:
                        recs = []
                        for item in lines[lines.index(line)+1:]:
                            if item.strip().startswith('-'):
                                recs.append(item.strip()[1:].strip())
                        result_data['recommendations'] = recs
                        break

                return render_template(
                    'interactive_result.html',
                    **result_data,
                    original_filename=filename
                )

            except Exception as e:
                return render_template(
                    'error.html',
                    error_message=f"Analysis failed: {str(e)}",
                    recovery_tip="Try uploading a clearer image of the crop leaves."
                )

    return render_template('detect.html')


'''
# Configure Gemini AI (should be at module level, not in route)
gemini_model = None
try:
    GEMINI_API_KEY = os.getenv('AIzaSyDiR28L-K2bRCHhZK4gye6MbHpeXZ62cpU', '').strip()
    if not GEMINI_API_KEY:
        # fallback if you want a default key (not recommended for production!)
        GEMINI_API_KEY = "AIzaSyDiR28L-K2bRCHhZK4gye6MbHpeXZ62cpU"

    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash')
except Exception as e:
    print(f"Failed to initialize Gemini: {e}")
    gemini_model = None


@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        # Validate file upload
        if 'file' not in request.files:
            flash('No file uploaded')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if not file.content_type.startswith('image/'):
            flash('Please upload an image file')
            return redirect(request.url)

        if file and gemini_model:
            try:
                # Secure file handling
                filename = secure_filename(file.filename)
                upload_dir = os.path.join('static', 'uploads')
                os.makedirs(upload_dir, exist_ok=True)
                img_path = os.path.join(upload_dir, filename)
                file.save(img_path)

                # Convert image to bytes for Gemini
                img = Image.open(img_path)
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG')
                img_bytes = img_byte_arr.getvalue()

                # Prompt
                prompt = """Analyze this crop image and provide detailed information in MARKDOWN format:

**Crop Name:** [Identify the crop species]
**Health Status:** [Healthy/Diseased/Stressed]
**Disease/Issue:** [Specific disease or problem if any]
**Confidence Level:** [High/Medium/Low]

**Recommendations:**
- [Treatment options]
- [Prevention methods]
- [Care instructions]

**Additional Notes:**
[Any other relevant observations]

Include appropriate emojis for better readability."""

                # Call Gemini
                response = gemini_model.generate_content([
                    {"role": "user", "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": img_bytes}}
                    ]}
                ])

                # Default result
                result_data = {
                    'crop': 'Unknown',
                    'status': 'Unknown',
                    'disease': 'None detected',
                    'confidence': 'N/A',
                    'recommendations': [],
                    'notes': '',
                    'image': img_path,
                    'status_icon': '❓'
                }

                if response.text:
                    analysis = response.text
                    result_data['raw_analysis'] = analysis  # keep raw output for debugging

                    # Extract info from Markdown
                    result_data['crop'] = extract_markdown_section(analysis, 'Crop Name')
                    result_data['status'] = extract_markdown_section(analysis, 'Health Status')
                    result_data['disease'] = extract_markdown_section(analysis, 'Disease/Issue')
                    result_data['confidence'] = extract_markdown_section(analysis, 'Confidence Level')
                    result_data['recommendations'] = extract_list_items(analysis, 'Recommendations')
                    result_data['notes'] = extract_markdown_section(analysis, 'Additional Notes')

                    # Set emoji
                    status_lower = result_data['status'].lower()
                    if 'healthy' in status_lower:
                        result_data['status_icon'] = '✅'
                    elif 'diseased' in status_lower:
                        result_data['status_icon'] = '⚠️'
                    elif 'stressed' in status_lower:
                        result_data['status_icon'] = '🌡️'

                return render_template('interactive_result.html', **result_data)

            except Exception as e:
                # Clean up file if error occurs
                if os.path.exists(img_path):
                    os.remove(img_path)
                return render_template(
                    'error.html',
                    error_message="Analysis failed. Please try again.",
                    recovery_tip="Try uploading a clearer image of the crop leaves"
                )
        else:
            flash('AI service is currently unavailable')
            return redirect(request.url)

    return render_template('detect.html')


# --- Helper functions ---

def extract_markdown_section(text, header):
    """Extract content after a markdown header like **Header:**"""
    marker = f'**{header}:**'
    if marker in text:
        return text.split(marker, 1)[1].split('\n')[0].strip()
    return ''


def extract_list_items(text, header):
    """Extract markdown list items after a header"""
    marker = f'**{header}:**'
    if marker in text:
        section = text.split(marker, 1)[1].split('\n\n')[0]
        return [line[2:].strip() for line in section.split('\n') if line.startswith('- ')]
    return [] '''


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('Logged out successfully.')
    return redirect(url_for('landing'))
    


@app.route('/rotation', methods=['GET', 'POST'])
def rotation():
    model = joblib.load("crop_rotation_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")

    if 'user_id' in session:
        if request.method == 'POST':
            try:
                # Get user inputs
                crop_name = request.form['crop_name']
                n = float(request.form['n'])
                p = float(request.form['p'])
                k = float(request.form['k'])
                soil_type = request.form['soil_type']
                season = request.form['season']
                temperature = float(request.form['temperature'])
                rainfall = float(request.form['rainfall'])
                humidity = float(request.form['humidity'])

                # Encode categorical inputs
                encoded_crop_name = label_encoders['Crop Name'].transform([crop_name])[0]
                encoded_soil_type = label_encoders['Soil Type'].transform([soil_type])[0]
                encoded_season = label_encoders['Season'].transform([season])[0]

                # Prepare input for prediction
                input_data = np.array([[encoded_crop_name, n, p, k, encoded_soil_type, encoded_season, temperature, rainfall, humidity]])

                # Predict next crop
                predicted_crop_encoded = model.predict(input_data)[0]
                predicted_next_crop = label_encoders['Preferred Next Crop'].inverse_transform([predicted_crop_encoded])[0]

                # Insert into DB
                with get_db_cursor(commit=True) as cursor:
                    insert_query = """
                        INSERT INTO crop_rotations 
                        (user_id, crop_name, n, p, k, soil_type, season, temperature, rainfall, humidity, predicted_next_crop)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(insert_query, (
                        session['user_id'], crop_name, n, p, k, soil_type, season,
                        temperature, rainfall, humidity, predicted_next_crop
                    ))

                # Normalize crop name for image matching
                predicted_next_crop_lower = predicted_next_crop.lower().replace(" ", "")

                # Crop image dictionary
                crop_images = {
                    'rice': 'crop_images/Rice.png',
                    'wheat': 'crop_images/wheat.png',
                    'maize': 'crop_images/maize.jpeg',
                    'corn': 'crop_images/maize.jpeg',
                    'apple': 'crop_images/apple.jpg',
                    'barley': 'crop_images/Barley.jpg',
                    'banana': 'crop_images/banana.jpeg',
                    'blackgram': 'crop_images/blackgram.jpg',
                    'chickpea': 'crop_images/chickpea.jpg',
                    'chickpea': 'crop_images/chickpea.jpg',
                    'chili': 'crop_images/chili.png',
                    'coconut': 'crop_images/coconut.jpg',
                    'coffee': 'crop_images/coffee.jpg',
                    'cotton': 'crop_images/cotton.jpeg',
                    'grapes': 'crop_images/grapes.jpeg',
                    'jute': 'crop_images/jute.jpeg',
                    'kidneybeans': 'crop_images/kidneybeans.jpeg',
                    'kidneybeans': 'crop_images/kidneybeans.jpeg',
                    'lentil': 'crop_images/lentil.jpeg',
                    'lentils': 'crop_images/lentil.jpeg',
                    'mango': 'crop_images/mango.jpeg',
                    'mothbeans': 'crop_images/mothbeans.jpeg',
                    'mungbeans': 'crop_images/mungbeans.jpeg',
                    'muskmelon': 'crop_images/muskmelon.jpeg',
                    'orange': 'crop_images/orange.jpeg',
                    'papaya': 'crop_images/papaya.jpeg',
                    'pigeonpeas': 'crop_images/pigeonpeas.jpeg',
                    'pomegranate': 'crop_images/pomergranate.jpg',
                    'tomato': 'crop_images/Tomato.png',
                    'watermelon': 'crop_images/watermelon.jpg',
                    'sugarcane': 'crop_images/sugarcane.png',
                    'soybean': 'crop_images/Soybean.png'
                }

                # Get image path or default
                crop_image = crop_images.get(predicted_next_crop_lower, 'crop_images/default.jpg')
                image_path = os.path.join('static', crop_image)
                if not os.path.exists(image_path):
                    crop_image = 'crop_images/default.jpg'

                return render_template('rotation_result.html',
                                       next_crop=predicted_next_crop,
                                       crop_image=crop_image)

            except Exception as e:
                return render_template('rotation.html', error="❌ Error: Invalid input values or unrecognized crop/season/soil type.")
        
        return render_template('rotation.html')

    return redirect(url_for('login'))
@app.route('/create_post', methods=['POST'])
def create_post():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    try:
        with get_db_cursor(commit=True) as cursor:
            user_id = session['user_id']
            content = request.form.get('content')
            image = request.files.get('image')

            image_filename = None
            if image and allowed_file(image.filename):
                image_filename = secure_filename(image.filename)
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
                image.save(image_path)

            cursor.execute(
                "INSERT INTO posts (user_id, content, image) VALUES (%s, %s, %s)", 
                (user_id, content, image_filename)
            )

        return redirect(url_for('community'))
    
    except Exception as e:
        print(f"Database error: {e}")
        flash("An error occurred while creating the post.")
        return redirect(url_for('community'))

@app.route('/community')
def community():
    try:
        with get_db_cursor() as cursor:
            cursor.execute("""
                SELECT posts.id, users.name, posts.content, posts.image, posts.created_at, 
                       COALESCE(users.profile_pic, 'default_profile.png') AS profile_picture,
                       (SELECT COUNT(*) FROM likes WHERE likes.post_id = posts.id) AS like_count
                FROM posts 
                JOIN users ON posts.user_id = users.id 
                ORDER BY posts.created_at DESC
            """)
            posts = cursor.fetchall()

            # Fetch comments for each post
            comments = {}
            for post in posts:
                cursor.execute("""
                    SELECT comments.comment, users.name, users.profile_pic 
                    FROM comments 
                    JOIN users ON comments.user_id = users.id 
                    WHERE comments.post_id = %s
                    ORDER BY comments.created_at ASC
                """, (post[0],))
                comments[post[0]] = cursor.fetchall()

        return render_template('community.html', posts=posts, comments=comments)
    
    except Exception as e:
        print(f"Database error: {e}")
        flash("An error occurred while loading the community page.")
        return redirect(url_for('home'))

@app.route('/like/<int:post_id>', methods=['POST'])
def like_post(post_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    
    try:
        with get_db_cursor(commit=True) as cursor:
            # Check if the user already liked the post
            cursor.execute("SELECT * FROM likes WHERE user_id = %s AND post_id = %s", (user_id, post_id))
            existing_like = cursor.fetchone()

            if not existing_like:
                cursor.execute("INSERT INTO likes (user_id, post_id) VALUES (%s, %s)", (user_id, post_id))

        return redirect(url_for('community'))
    
    except Exception as e:
        print(f"Database error: {e}")
        flash("An error occurred while liking the post.")
        return redirect(url_for('community'))

@app.route('/comment/<int:post_id>', methods=['POST'])
def comment_post(post_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    comment_text = request.form['comment']

    if comment_text.strip():
        try:
            with get_db_cursor(commit=True) as cursor:
                cursor.execute("INSERT INTO comments (user_id, post_id, comment) VALUES (%s, %s, %s)", 
                               (user_id, post_id, comment_text))
            return redirect(url_for('community'))
        
        except Exception as e:
            print(f"Database error: {e}")
            flash("An error occurred while adding the comment.")
            return redirect(url_for('community'))

import os

@app.route('/delete_post/<int:post_id>', methods=['POST'])
def delete_post(post_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    try:
        with get_db_cursor(commit=True) as cursor:
            # Fetch the image filename before deleting the post
            cursor.execute("SELECT content FROM posts WHERE id = %s", (post_id,))
            post = cursor.fetchone()

            if post and post[0]:  # If the post exists and has an image
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], post[0])
                if os.path.exists(image_path):  # Check if the file exists
                    os.remove(image_path)  # Delete the file

            # Delete the post from the database
            cursor.execute("DELETE FROM posts WHERE id = %s", (post_id,))

        return redirect(url_for('community'))
    
    except Exception as e:
        print(f"Database error: {e}")
        flash("An error occurred while deleting the post.")
        return redirect(url_for('community'))


@app.route('/my_posts')
def my_posts():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']

    try:
        with get_db_cursor() as cursor:
            # Fetch the logged-in user's posts
            cursor.execute("""
                SELECT posts.id, users.name, posts.content, posts.image, posts.created_at, 
                       COALESCE(users.profile_pic, 'default_profile.png') AS profile_picture,
                       (SELECT COUNT(*) FROM likes WHERE likes.post_id = posts.id) AS like_count
                FROM posts 
                JOIN users ON posts.user_id = users.id 
                WHERE posts.user_id = %s
                ORDER BY posts.created_at DESC
            """, (user_id,))
            
            user_posts = cursor.fetchall()

            # Fetch comments for each post
            user_comments = {}
            for post in user_posts:
                cursor.execute("""
                    SELECT comments.comment, users.name, users.profile_pic 
                    FROM comments 
                    JOIN users ON comments.user_id = users.id 
                    WHERE comments.post_id = %s
                    ORDER BY comments.created_at ASC
                """, (post[0],))
                user_comments[post[0]] = cursor.fetchall()

        return render_template('my_posts.html', posts=user_posts, comments=user_comments)
    
    except Exception as e:
        print(f"Database error: {e}")
        flash("An error occurred while loading your posts.")
        return redirect(url_for('home'))


if __name__ == '__main__': 
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
