import os
import pickle
import re
from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Ensure that NLTK resources are downloaded
nltk.download('stopwords')

app = Flask(__name__)

# Get the directory of the current script
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, 'log_reg.pkl')
vectorizer_path = os.path.join(current_dir, 'vectorizer.pkl')

# Load the pre-trained model
with open(model_path, 'rb') as file:
    model = pickle.load(file)

port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            news_article = request.form['news_article']
            stemmed_article = stemming(news_article)

            # Load the vectorizer
            with open(vectorizer_path, 'rb') as file:
                vectorizer = pickle.load(file)
            
            article_vectorized = vectorizer.transform([stemmed_article])
            prediction = model.predict(article_vectorized)

            result = 'Fact' if prediction[0] == 0 else 'Fake'
            return render_template('home.html', prediction_text='The news is {}'.format(result))

        except Exception as e:
            return render_template('home.html', prediction_text='Error: {}'.format(e)), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
