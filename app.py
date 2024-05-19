from flask import Flask, render_template, request
import pickle
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer 
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Get the directory of the current script
current_dir = os.path.dirname(__file__)
# Construct the path to the model file
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
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
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
        news_article = request.form['news_article']
        stemmed_article = stemming(news_article)
        
        # Load the vectorizer used during training
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        
        article_vectorized = vectorizer.transform([stemmed_article])
        prediction = model.predict(article_vectorized)

        if prediction[0] == 0:
            result = 'Fact'
        else:
            result = 'Fake'

        return render_template('home.html', prediction_text='The news is {}'.format(result))

if __name__ == '__main__':
    app.run(debug=True)
