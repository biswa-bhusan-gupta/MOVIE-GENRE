from flask import Flask,request,render_template
import pickle
import re
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


app=Flask(__name__)

CV=pickle.load(open('CV_Transform.pkl','rb'))
NB=pickle.load(open('MovieGenre.pkl','rb'))


@app.route("/")
def Index():
    return render_template("Index.html")

@app.route("/Predict",methods=['POST'])
def Predict():
    if request.method == 'POST':
        Script = request.form['Script']
        Script=re.sub('[^a-zA-Z]',' ',Script)
        Script=Script.lower()
        Script=Script.split()

        Lemmatizer=WordNetLemmatizer()
        Script=[Lemmatizer.lemmatize(Word) for Word in Script if Word not in set(stopwords.words('english'))]

        Script=' '.join(Script)

        Temp=CV.transform([Script]).toarray()
        Prediction=NB.predict(Temp)[0]


        return render_template('Predict.html',prediction=Prediction)

if __name__ == '__main__':
    app.run(debug=True)
