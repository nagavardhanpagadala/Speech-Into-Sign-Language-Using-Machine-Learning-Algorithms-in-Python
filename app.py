from flask import Flask, render_template, redirect, url_for, request, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import speech_recognition as sr
from gtts import gTTS
import os
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secure_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Create required directories
os.makedirs(os.path.join(app.root_path, 'static', 'temp'), exist_ok=True)
os.makedirs(os.path.join(app.root_path, 'static', 'ISL_Gifs'), exist_ok=True)

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Database Models
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    histories = db.relationship('SpeechHistory', backref='user', lazy=True)

class SpeechHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    text = db.Column(db.String(500), nullable=False)
    gif_path = db.Column(db.String(200))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# ISL GIF phrases list
isl_gif_phrases = [
    'any questions', 'are you angry', 'are you busy', 'are you hungry', 'are you sick', 'be careful',
    'can we meet tomorrow', 'did you book tickets', 'did you finish homework', 'do you go to office',
    'do you have money', 'do you want something to drink', 'do you want tea or coffee', 'do you watch TV',
    'dont worry', 'flower is beautiful', 'good afternoon', 'good evening', 'good morning', 'good night',
    'good question', 'had your lunch', 'happy journey', 'hello what is your name', 'how many people are there in your family',
    'i am a clerk', 'i am bore doing nothing', 'i am fine', 'i am sorry', 'i am thinking', 'i am tired',
    'i dont understand anything', 'i go to a theatre', 'i love to shop', 'i had to say something but i forgot',
    'i have headache', 'i like pink colour', 'i live in nagpur', 'lets go for lunch', 'my mother is a homemaker',
    'my name is john', 'nice to meet you', 'no smoking please', 'open the door', 'please call me later',
    'please clean the room', 'please give me your pen', 'please use dustbin dont throw garbage', 'please wait for sometime',
    'shall I help you', 'shall we go together tomorrow', 'sign language interpreter', 'sit down', 'stand up',
    'take care', 'there was traffic jam', 'wait I am thinking', 'what are you doing', 'what is the problem',
    'what is todays date', 'what is your father do', 'what is your job', 'what is your mobile number', 'what is your name',
    'whats up', 'when is your interview', 'when we will go', 'where do you stay', 'where is the bathroom',
    'where is the police station', 'you are wrong', 'address', 'agra', 'ahemdabad', 'all', 'april', 'assam', 'august',
    'australia', 'badoda', 'banana', 'banaras', 'banglore', 'bihar', 'bridge', 'cat', 'chandigarh', 'chennai',
    'christmas', 'church', 'clinic', 'coconut', 'crocodile', 'dasara', 'deaf', 'december', 'deer', 'delhi', 'dollar',
    'duck', 'february', 'friday', 'fruits', 'glass', 'grapes', 'gujrat', 'hello', 'hindu', 'hyderabad', 'india',
    'january', 'jesus', 'job', 'july', 'karnataka', 'kerala', 'krishna', 'litre', 'mango', 'may', 'mile', 'monday',
    'mumbai', 'museum', 'muslim', 'nagpur', 'october', 'orange', 'pakistan', 'pass', 'police station', 'post office',
    'pune', 'punjab', 'rajasthan', 'ram', 'restaurant', 'saturday', 'september', 'shop', 'sleep', 'southafrica',
    'story', 'sunday', 'tamil nadu', 'temperature', 'temple', 'thursday', 'toilet', 'tuesday', 'village', 'wednesday',
    'weight', 'what', 'where', 'you', 'your','capital a',
'capital b',
'capital c',
'capital d',
'capital e',
'capital f',
'capital g',
'capital h',
'capital i',
'capital j',
'capital k',
'capital l',
'capital m',
'capital n',
'capital o',
'capital p',
'capital q',
'capital r',
'capital s',
'capital t',
'capital u',
'capital v',
'capital w',
'capital x',
'capital y',
'capital z'
]

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
gif_phrase_vectors = vectorizer.fit_transform(isl_gif_phrases)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
# Core functionality for matching and speech processing
def find_best_match(text):
    """Enhanced matching algorithm combining multiple approaches"""
    try:
        # Convert text to lowercase for matching
        text = text.lower().strip()
        
        # Direct match
        for phrase in isl_gif_phrases:
            if phrase.lower() in text:
                return phrase

        # Partial match
        words = text.split()
        for phrase in isl_gif_phrases:
            phrase_words = phrase.lower().split()
            if any(word in phrase_words for word in words):
                return phrase

        # Cosine similarity as fallback
        text_vector = vectorizer.transform([text])
        similarities = cosine_similarity(text_vector, gif_phrase_vectors)
        best_match_index = similarities.argmax()
        similarity_score = similarities[0][best_match_index]

        if similarity_score > 0.3:  # Threshold for acceptable match
            return isl_gif_phrases[best_match_index]
        
        return None
    except Exception as e:
        print(f"Error in find_best_match: {str(e)}")
        return None

# Speech Recognition Routes
@app.route('/run_assistant', methods=['GET'])
@login_required
def run_assistant():
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print("Listening...")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            print("Processing speech...")
            
            text = recognizer.recognize_google(audio).lower()
            print(f"Recognized: {text}")

            # Find matching GIF
            best_match = find_best_match(text)
            gif_path = None

            if best_match:
                gif_filename = f'{best_match}.gif'
                gif_path = url_for('static', filename=f'ISL_Gifs/{gif_filename}')
                
                # Verify GIF exists
                full_path = os.path.join(app.root_path, 'static', 'ISL_Gifs', gif_filename)
                if not os.path.exists(full_path):
                    print(f"GIF not found: {full_path}")
                    gif_path = None

            # Save to history
            history = SpeechHistory(
                user_id=current_user.id,
                text=text,
                gif_path=gif_path
            )
            db.session.add(history)
            db.session.commit()

            return jsonify({
                'success': True,
                'text': text,
                'gif_path': gif_path,
                'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            }), 200

    except sr.WaitTimeoutError:
        return jsonify({'error': 'Listening timeout. Please try again.'}), 400
    except sr.UnknownValueError:
        return jsonify({'error': 'Could not understand audio. Please speak clearly.'}), 400
    except sr.RequestError as e:
        return jsonify({'error': f'Speech recognition service error: {str(e)}'}), 500
    except Exception as e:
        print(f"Error in run_assistant: {str(e)}")
        db.session.rollback()
        return jsonify({'error': 'An unexpected error occurred. Please try again.'}), 500

# Text Processing Route
@app.route('/process_text', methods=['POST'])
@login_required
def process_text():
    try:
        data = request.get_json()
        text = data.get('text', '').lower()

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Find matching GIF
        best_match = find_best_match(text)
        gif_path = None

        if best_match:
            gif_filename = f'{best_match}.gif'
            gif_path = url_for('static', filename=f'ISL_Gifs/{gif_filename}')
            
            # Verify GIF exists
            full_path = os.path.join(app.root_path, 'static', 'ISL_Gifs', gif_filename)
            if not os.path.exists(full_path):
                gif_path = None

        # Generate speech audio
        try:
            tts = gTTS(text=text, lang='en')
            audio_filename = f'speech_{current_user.id}_{int(datetime.utcnow().timestamp())}.mp3'
            audio_path = os.path.join(app.root_path, 'static', 'temp', audio_filename)
            tts.save(audio_path)
            audio_url = url_for('static', filename=f'temp/{audio_filename}')
        except Exception as e:
            print(f"TTS Error: {str(e)}")
            audio_url = None

        # Save to history
        history = SpeechHistory(
            user_id=current_user.id,
            text=text,
            gif_path=gif_path
        )
        db.session.add(history)
        db.session.commit()

        return jsonify({
            'success': True,
            'text': text,
            'gif_path': gif_path,
            'audio_path': audio_url,
            'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        }), 200

    except Exception as e:
        print(f"Error in process_text: {str(e)}")
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# Utility function for cleaning up temporary files
def cleanup_temp_files():
    temp_dir = os.path.join(app.root_path, 'static', 'temp')
    current_time = datetime.utcnow().timestamp()
    
    for filename in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, filename)
        if os.path.isfile(file_path):
            # Remove files older than 1 hour
            if current_time - os.path.getctime(file_path) > 3600:
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error removing file {filename}: {e}")
# Authentication and Main Routes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
@login_required
def predict():
    return render_template('predict.html')

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        
        # Enhanced validation
        if not all(key in data for key in ['username', 'email', 'password']):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Validate email format
        if '@' not in data['email'] or '.' not in data['email']:
            return jsonify({'error': 'Invalid email format'}), 400
        
        # Validate username length
        if len(data['username']) < 3:
            return jsonify({'error': 'Username must be at least 3 characters long'}), 400
        
        # Validate password strength
        if len(data['password']) < 6:
            return jsonify({'error': 'Password must be at least 6 characters long'}), 400
        
        # Check existing users
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'error': 'Email already registered'}), 400
        
        if User.query.filter_by(username=data['username']).first():
            return jsonify({'error': 'Username already taken'}), 400
        
        # Create new user
        hashed_password = generate_password_hash(data['password'])
        new_user = User(
            username=data['username'],
            email=data['email'],
            password=hashed_password
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Registration successful',
            'user_id': new_user.id
        }), 201
        
    except Exception as e:
        print(f"Registration error: {str(e)}")
        db.session.rollback()
        return jsonify({'error': 'Registration failed. Please try again.'}), 500

@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        
        # Validate required fields
        if not all(key in data for key in ['email', 'password']):
            return jsonify({'error': 'Missing required fields'}), 400
        
        user = User.query.filter_by(email=data['email']).first()
        
        if user and check_password_hash(user.password, data['password']):
            login_user(user, remember=True)
            return jsonify({
                'success': True,
                'message': 'Login successful',
                'username': user.username,
                'redirect_url': url_for('predict')
            }), 200
        
        return jsonify({'error': 'Invalid email or password'}), 401
        
    except Exception as e:
        print(f"Login error: {str(e)}")
        return jsonify({'error': 'Login failed. Please try again.'}), 500

@app.route('/logout')
@login_required
def logout():
    try:
        logout_user()
        return jsonify({
            'success': True,
            'message': 'Logout successful',
            'redirect_url': url_for('index')
        }), 200
    except Exception as e:
        print(f"Logout error: {str(e)}")
        return jsonify({'error': 'Logout failed'}), 500

@app.route('/get_history')
@login_required
def get_history():
    try:
        # Get user's speech history, ordered by timestamp
        history = SpeechHistory.query.filter_by(user_id=current_user.id)\
            .order_by(SpeechHistory.timestamp.desc())\
            .limit(10)\
            .all()
        
        history_data = [{
            'id': h.id,
            'text': h.text,
            'gif_path': h.gif_path,
            'timestamp': h.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        } for h in history]
        
        return jsonify({
            'success': True,
            'history': history_data
        }), 200
        
    except Exception as e:
        print(f"History fetch error: {str(e)}")
        return jsonify({'error': 'Failed to fetch history'}), 500

# Error Handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({
        'error': 'Resource not found',
        'message': 'The requested resource could not be found on the server.'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred. Please try again later.'
    }), 500

# Session Management
@app.before_request
def before_request():
    if current_user.is_authenticated:
        # Cleanup old temporary files before processing new requests
        cleanup_temp_files()

# Application Initialization
def init_app():
    with app.app_context():
        # Create all database tables
        db.create_all()
        
        # Create required directories if they don't exist
        os.makedirs(os.path.join(app.root_path, 'static', 'temp'), exist_ok=True)
        os.makedirs(os.path.join(app.root_path, 'static', 'ISL_Gifs'), exist_ok=True)
        
        # Initial cleanup
        cleanup_temp_files()

if __name__ == '__main__':
    init_app()
    app.run(debug=True, host='0.0.0.0', port=5000)