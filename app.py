from flask import Flask, render_template, request,  send_file, make_response, url_for, session, redirect, jsonify
import requests
import os
import io
from io import BytesIO
import re
from ai import  predict_df, from_SMILE_to_mol, extract_metabolite_smiles
from PIL import Image
import json
import pandas as pd
import base64

UPLOAD_FOLDER = 'upload_folder'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

@app.route('/')
def home():
    if 'df' in globals():
        global df
        del df
    if 'stats' in globals():
        global stats
        del stats
    if 'count' in globals():
        global count
        del count
    return render_template('index.html', is_csv_pred= False, show_table = False)

###### Original function
# @app.route('/predict', methods=['POST'])
# def predict():
#     reaction_smile = request.form['reaction_smile']
#     global rxn_smile
#     rxn_smile = reaction_smile
#     # Call your AI prediction function here
#     prediction = predict_number(reaction_smile)
#     global ec_predicted
#     ec_predicted = prediction
#     return render_template('index.html', one_prediction = True, prediction=prediction)


###### Function with REST API
@app.route('/predict', methods=['POST'])
def predict():
    #reaction_smile = request.form['reaction_smile']

    user_input = request.json.get('input')
    print("user_input", user_input)
    response = requests.post('http://localhost:5001/process', json={'input': user_input})

    result = response.json().get('result')
    print("result", result)

    global rxn_smile
    rxn_smile = user_input

    global ec_predicted
    ec_predicted = result

    #return render_template('index.html', one_prediction = True, prediction=result)
    return jsonify({'result': result})



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/image')
def show_image():
    if 'rxn_smile' in globals():
        global rxn_smile
        global ec_predicted
        
        prediction = ec_predicted
        reactants_smile, products_smile =  extract_metabolite_smiles(rxn_smile)
        print("reactants_smile", reactants_smile, "products_smile", products_smile)
        print("GOT USER INPUT")
        if from_SMILE_to_mol(reactants_smile, products_smile) is None:
            return send_file("images/No_SMILE.png", mimetype='image/png')
        
        reactants, products = from_SMILE_to_mol(reactants_smile, products_smile)
        print("reactants", reactants, "products", products)

        reactants_img = []
        for i in range(len(reactants)):
            smile = reactants[i]
            image_io = io.BytesIO()
            smile.save(image_io, format='PNG')
            image_io.seek(0)
            
            reactants_img.append(base64.b64encode(image_io.getvalue()).decode('utf-8'))
            
        products_img = []
        for j in range(len(products)):
            smile = products[j]
            image_io = io.BytesIO()
            smile.save(image_io, format='PNG')
            image_io.seek(0)
            
            products_img.append(base64.b64encode(image_io.getvalue()).decode('utf-8'))
        del rxn_smile
        del ec_predicted
        # send_file(image_io, mimetype='image/png')
        return render_template('index.html',  one_prediction =True, prediction = prediction, is_image_shown = True, reactants=reactants_img, len_reactant = len(reactants_img), products=products_img, len_product = len(products_img)) 
    else:
        print("NO USER INPUT")
        return send_file("images/no_user_input.png", mimetype='image/png')

@app.route('/table')
def table():
    global df
    global stats
    global count
    if df is None:
        return "DataFrame not found. Please create it first."
    table_dataframe = df.copy(deep=True)

    table_dataframe = table_dataframe[['First_Prediction', 'First_Confidence_score', 'First_Confidence_score_categorical', 'Second_Prediction', 'Second_Confidence_score', 'Second_Confidence_score_categorical']]
    table_dataframe = table_dataframe.rename(columns={'First_Prediction': 'Prediction 1', 'First_Confidence_score': 'CS 1', 'First_Confidence_score_categorical': 'Category 1', 'Second_Prediction': 'Prediction 2', 'Second_Confidence_score': 'CS 2', 'Second_Confidence_score_categorical': 'Category 2'})
    page = request.args.get('page', 1, type=int)
    per_page = 5
    total = len(table_dataframe)
    start = (page - 1) * per_page
    end = start + per_page
    paginated_df = table_dataframe.iloc[start:end]
    return render_template('index.html', show_table = True, tables=[paginated_df.to_html(classes='data')], titles=paginated_df.columns.values, page=page, total=total, per_page=per_page, statistics = stats, count_first_ec = count)

@app.route('/no_table',  methods=['POST'])
def hide_table():
    global stats
    global count
    return render_template('index.html',is_csv_pred= True, statistics = stats, count_first_ec = count)


###### Original function
# @app.route('/predict_csv', methods=['POST', 'GET'])
# def predict_csv():
#     if 'file' not in request.files:
#         print("NO FILE PART")
#         return render_template('index.html', csv_predictions=[], error="No file part")
    
#     file = request.files['file']
#     if file.filename == '':
#         print("NO SELECTED FILE")
#         return render_template('index.html', csv_predictions=[], error="No selected file")
    
#     if file and allowed_file(file.filename):
#         print("FILE FOUND")
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#         file.save(filepath)
        
#         # Generate the output DataFrame
#         global df
#         global stats

#         predictions_df, statistics = predict_df(filepath)

#         df = predictions_df
#         stats = statistics
#         # Save the processed DataFrame to a BytesIO object for download purposes
#         output = BytesIO()
#         predictions_df.to_csv(output, index=False)
#         output.seek(0)
#         # Store the output in the session or a global variable
#         global processed_csv
#         processed_csv = output
#         return render_template('index.html', 
#                                is_csv_pred= True,  
#                                statistics = stats)


###### Function with REST API


@app.route('/predict_csv', methods=['POST', 'GET'])
def predict_csv():
    if 'file' not in request.files:
        print("NO FILE PART")
        return render_template('index.html', csv_predictions=[], error="No file part")
    
    file = request.files['file']
    if file.filename == '':
        print("NO SELECTED FILE")
        return render_template('index.html', csv_predictions=[], error="No selected file")
    
    if file and allowed_file(file.filename):
        print("FILE FOUND")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Generate the output DataFrame
        with open(filepath, 'rb') as f:
            response = requests.post('http://localhost:5001/process_csv', files={'input_file': f})
      
        result_csv = response.json().get('result')
        predictions_df = pd.read_csv(io.StringIO(result_csv))
        statistics = response.json().get('statistics')
        count_ec = response.json().get('count_first_ec')
        
        print("predictions_df", predictions_df, "statistics", statistics, "count_ec", count_ec)

        global df
        global stats
        global count        
        df = predictions_df
        stats = statistics
        count = count_ec
        # Save the processed DataFrame to a BytesIO object for download purposes
        output = BytesIO()
        predictions_df.to_csv(output, index=False)
        output.seek(0)
        # Store the output in the session or a global variable
        global processed_csv
        processed_csv = output
        return render_template('index.html', 
                               is_csv_pred= True,  
                               statistics = stats,
                               count_first_ec = count)
    
@app.route('/download', methods=['GET'])
def download_file():
    if 'processed_csv' not in globals():
        return 'No file to download'
    global processed_csv
    response = make_response(processed_csv.getvalue())
    response.headers['Content-Disposition'] = 'attachment; filename=processed_output.csv'
    response.headers['Content-Type'] = 'text/csv'
    del processed_csv
    return response





if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    app.run(port=5000, debug=True)