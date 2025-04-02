from flask import Flask, render_template, request, url_for
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))


@app.route('/')
def Home():
    dict_jobtitle = pickle.load(open('jobtitle.pkl','rb'))
    dict_jobtitle_list = list(dict_jobtitle.keys())
    return render_template('index.html',options = dict_jobtitle_list)

@app.route('/predict',methods=['POST'])
def Predict():
    dict_jobtitle = pickle.load(open('jobtitle.pkl','rb'))
    dict_educationlevel = pickle.load(open('educationlevel.pkl','rb'))
    dict_jobtitle_list = list(dict_jobtitle.keys())
    age = int(request.form['selected_option_age'])
    yoe = int(request.form['selected_option_yoe'])
    edu_level = request.form['selected_option_edu_level']
    job_title = request.form['selected_option_job_title']
    edu_level_1 = int(dict_educationlevel[edu_level])
    job_title_1 = int(dict_jobtitle[job_title])
    features = [age,yoe,edu_level_1,job_title_1]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    output = round(prediction[0])
    
    return render_template('index.html', options = dict_jobtitle_list, prediction_text = "Salary should be $ {}".format(output))

if __name__=='__main__':
    app.run(debug=True)