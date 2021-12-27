
#!/usr/bin/env python
# from: https://blog.csdn.net/qq_36387683/article/details/85112867

import uuid
from flask import Flask
from flask import Flask,render_template,request,redirect,url_for,send_file
from werkzeug.utils import secure_filename
import os
import algorithm_test
from flask_sessionstore import Session
from flask_session_captcha import FlaskSessionCaptcha

app = Flask(__name__)
app.config["SECRET_KEY"] = uuid.uuid4()
app.config['CAPTCHA_ENABLE'] = True
app.config['CAPTCHA_LENGTH'] = 5
app.config['CAPTCHA_WIDTH'] = 160
app.config['CAPTCHA_HEIGHT'] = 60
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite://'
app.config['SESSION_TYPE'] = 'sqlalchemy'

Session(app)
captcha = FlaskSessionCaptcha(app)



def test():
  fin = open(r'templates\upload.html', 'r')
  fout = open(r'templates\test.html', 'w')

  for line in fin:
    if line.strip() == '</body>':
      fout.write('\t\t<h1>ceshi</h1>\n')
    fout.write(line)


#https://www.javaroad.cn/questions/6410
#https://blog.csdn.net/qq_44741960/article/details/106142305



@app.route('/', methods=['POST', 'GET'])
def upload():
    MODEL_CHOICE = {'adaboost':0, 'decision_tree':1, 'random_forest':2, 'logistic_regression':3, 'ridge_regression':4}
    task = {'train_lower_sensitivity': "0.0",
                  'train_lower_specificity': "0.0",
                  'train_lower_ppv': "0.0",
                  'train_lower_npv': "0.0",
                  'train_lower_cutoff': "0.0",
                  'train_upper_sensitivity': "0.0",
                  'train_upper_specificity': "0.0",
                  'train_upper_ppv': "0.0",
                  'train_upper_npv': "0.0",
                  'train_upper_cutoff': "0.0",
                  'train_lower_percent': "0.0",
                  'train_upper_percent': "0.0",
                  'train_num_parameter': "0.0",
                  'train_num_sample': "0.0",
                  'train_auroc': "0.0",
                  'test_lower_sensitivity': "0.0",
                  'test_lower_specificity': "0.0",
                  'test_lower_ppv': "0.0",
                  'test_lower_npv': "0.0",
                  'test_lower_cutoff': "0.0",
                  'test_upper_sensitivity': "0.0",
                  'test_upper_specificity': "0.0",
                  'test_upper_ppv': "0.0",
                  'test_upper_npv': "0.0",
                  'test_upper_cutoff': "0.0",
                  'test_lower_percent': "0.0",
                  'test_upper_percent': "0.0",
                  'test_num_parameter': "0.0",
                  'test_num_sample': "0.0",
                  'test_auroc': "0.0"
                  }

    url1 = {"link":""}
    #print(request.form)
    if request.method == 'POST':
        if request.method == "POST":
            if captcha.validate():
                return "success"
            else:
                return "fail"
        task["test"] = "no"
        print(request.form.to_dict().keys())
        keys = request.form.to_dict().keys()
        if "upload_file" in keys:
            f = request.files['file']
            basepath = os.path.dirname(__file__)  # 当前文件所在路径
            upload_path = os.path.join(basepath, r'static\uploads', secure_filename(f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
            f.save(upload_path)
            print(f.filename)

        if "select_table_submit" in keys:
            f = request.files['file']
            basepath = os.path.dirname(__file__)  # 当前文件所在路径
            upload_path = os.path.join(basepath, r'static\uploads',
                                       secure_filename(f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
            f.save(upload_path)
            print(f.filename)

            print(request.form.get("select_table"))
            print(request.files['file'].filename)
            task = algorithm_test.exe_model(model=MODEL_CHOICE[request.form.get("select_table")], data_path=upload_path, ratio=0.3, predicting_year=5)
            print(task)
            url1["link"] = "you may download output from: " + r"http://127.0.0.1:5000/download/Test.csv"
            download(r"D:\Code\WebPage_Test\Test.csv")
        if "run" in keys:
            print("test")


        return render_template('test_for_hcc.html',task=task,url1=url1)



        #test()
        #return render_template('test.html')
        #return redirect(url_for('upload'))

    return render_template('test_for_hcc.html',task=task, url1=url1)



@app.route('/download/<path:filename>',methods=['get'])
def download(filename):
    return send_file(filename,as_attachment=True,attachment_filename='Test.csv')





if __name__ == '__main__':
    app.run(debug=True)