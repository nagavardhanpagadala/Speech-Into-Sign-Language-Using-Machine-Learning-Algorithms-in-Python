# Speech-Into-Sign Language Using Machine Learning Algorithms in Python

## Project Overview
Speech-to-Sign Language Converter is a web application that bridges communication gaps between hearing and deaf communities. It converts spoken words and text input into Indian Sign Language (ISL) animations, making communication more accessible and inclusive.
![Screenshot 2025-01-07 123726](https://github.com/user-attachments/assets/f2a72833-59e6-433a-981a-9f45f88069b7)
![main interface](https://github.com/user-attachments/assets/d7880872-0cbb-4a2f-a69c-cf5e6120a00d)

## Features
- **Speech Recognition**: Real-time conversion of spoken words to text
- **Text Input**: Direct text input for conversion to sign language
- **ISL Animation**: Display of corresponding Indian Sign Language animations
- **Text-to-Speech**: Convert input text to audio for bilateral communication
- **User Authentication**: Secure login and registration system
- **History Tracking**: Save and view previous conversions
- **Responsive Design**: Works on both desktop and mobile devices

## Technical Stack
- **Backend**: Flask (Python)
- **Database**: SQLite with SQLAlchemy ORM
- **Authentication**: Flask-Login
- **Speech Processing**: 
  - SpeechRecognition for speech-to-text
  - gTTS (Google Text-to-Speech) for text-to-audio
- **Natural Language Processing**: 
  - NLTK for text processing
  - Scikit-learn for text similarity matching
- **Frontend**:
  - HTML/CSS/JavaScript
  - AJAX for asynchronous requests

## Prerequisites
- Python 3.8+
- pip package manager
- Virtual environment (recommended)
- Microphone access for speech recognition
- Speakers/headphones for audio output

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd speech-to-sign-converter
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Initialize the database:
```bash
python
>>> from app import app, db, init_app
>>> with app.app_context():
...     init_app()
```

5. Add ISL GIF files:
- Create directory: `static/ISL_Gifs`
- Add your ISL animation GIFs named according to the phrases in `isl_gif_phrases` list

## Running the Application
1. Start the Flask server:
```bash
python app.py
```

2. Access the application:
- Open your web browser
- Navigate to `http://localhost:5000`

## Usage
1. **Register/Login**:
   - Create a new account or login with existing credentials

2. **Speech to Sign**:
   - Click the microphone button
   - Speak clearly into your microphone
   - View the corresponding ISL animation

3. **Text to Sign**:
   - Type text in the input field
   - Submit to view the ISL animation
   - Listen to the audio playback

4. **View History**:
   - Access your previous translations
   - Review past conversations

## Project Structure
```
speech-to-sign-converter/
├── app.py                 # Main application file
├── requirements.txt       # Project dependencies
├── static/
│   ├── ISL_Gifs/         # Sign language animation GIFs
│   ├── temp/             # Temporary audio files
│   ├── css/              # Stylesheets
│   └── js/               # JavaScript files
├── templates/
│   ├── index.html        # Landing page
│   └── predict.html      # Main application page
└── users.db              # SQLite database
```

## Security Features
- Password hashing using Werkzeug
- Session management with Flask-Login
- CSRF protection
- Input validation and sanitization
- Secure file handling

## Error Handling
- Speech recognition timeout and errors
- Audio processing failures
- Database transaction errors
- File system errors
- Authentication failures

## Maintenance
- Automatic cleanup of temporary audio files
- Database session management
- Error logging and monitoring

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Acknowledgments
- Flask and related extensions
- Google Speech Recognition API
- NLTK and scikit-learn communities
- Indian Sign Language resources
