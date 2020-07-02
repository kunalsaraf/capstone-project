from flask import Flask, render_template, request, redirect
import os
import base64
import app.currency as currency_module 

# Flask config
app = Flask(__name__)
# app.secret_key = FLASK_SECRET_KEY
# app.config['WTF_CSRF_TIME_LIMIT'] = WTF_CSRF_TIME_LIMIT
UPLOAD_FOLDER_CURRENCY = './app/static/uploads/currency/'
UPLOAD_FOLDER_CAPTION = './app/static/uploads/caption/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

def rename_file(name):
    name,ext = os.path.splitext(name)
    return 'image'+ext

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/currency', methods=['GET', 'POST'])
def currency():
    if request.method == 'GET':
        return render_template('currency.html', result_generated = "")
    elif request.method == 'POST':
        file1 = request.files['file1']
        if(file1.filename != ""):
            file1.save(UPLOAD_FOLDER_CURRENCY+rename_file(file1.filename))
            filenameToBeSent = rename_file(file1.filename)
            currency_module.get_label(f'./app/static/uploads/currency/{filenameToBeSent}')
        else:
            file2 = request.form['file2']
            imgdata = base64.b64decode(file2[22:])
            filenameToBeSent = "image.png"
            with open(UPLOAD_FOLDER_CURRENCY+"image.png", "wb") as fh:
                fh.write(imgdata)
            currency_module.get_label('./app/static/uploads/currency/image.png')
        return render_template('currency.html', result_generated = filenameToBeSent)

@app.route('/caption', methods=['GET', 'POST'])
def caption():
    if request.method == 'GET':
        return render_template('caption.html')
    elif request.method == 'POST':
        file1 = request.files['file1']
        if(file1.filename != ""):
            file1.save(UPLOAD_FOLDER_CAPTION+rename_file(file1.filename))
        else:
            file2 = request.form['file2']
            imgdata = base64.b64decode(file2[22:])
            with open(UPLOAD_FOLDER_CAPTION+"image.png", "wb") as fh:
                fh.write(imgdata)
        return redirect('/caption')