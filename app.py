from flask import Flask, render_template, url_for, request
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle 
import content_based
import colab
# import hybrid_based

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
data=pd.read_csv('C:/Users/Aishwarya K/Desktop/book-recommendation-system-webapp-master/data.csv')


@app.route('/svd', methods=['GET', 'POST'])
def svd():
    if request.method == 'POST':
        input_data = request.form['input']
        output_data = colab.predict_function(input_data)
        p = []
        for i, prediction in enumerate(output_data):
            p.append('{no}. News article {iid} (predicted rating: {est})'.format(no= i+1,iid=prediction.iid,est = prediction.est))
        print(p)   
        return render_template('svd.html', output= p)
    else:
        return render_template('result.html')
    

@app.route('/',methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        input_data = int(request.form['input'])
        output=content_based.tf_idf_based_model(input_data,11,1,1,1)
        
        return render_template("home.html",output=output)
    else:
        return render_template('home.html')

@app.route('/svdIndex')
def svdIndex():
    return render_template('svd.html')

@app.route('/hybrid', methods = ['GET','POST'])
def hybrid():
    if request.method == 'POST':
        input_data = int(request.form['svd'])

        # res= hybrid_based.hybrid_recommendation(input_data)
        list=[7033,5170,6987,6472,5219,6784,5033,4838,6450,4853]
        rec=[]
        for i in list:
            rec.append('News article {iid}'.format(iid=i,)) 
        return render_template("hybrid.html",output=rec)
    else:
        return render_template('hybrid.html')


if __name__ == '__main__':
    app.run(debug=True,port=8001)
