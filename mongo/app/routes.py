from app import app
from flask import Flask
from flask import render_template
from flask import request,session
from app.forms import LoginForm
import os

app = Flask(__name__)
app.config["SECRET_KEY"] = "12345678"
APP_ROOT = os.path.dirname(os.path.abspath(__file__))   # refers to application_top
APP_STATIC_TXT = os.path.join(APP_ROOT, 'static') #设置一个专门的类似全局变量的东西

@app.route('/hello')
def  hello():
    return "Hello, World!"

@app.route("/login", methods=['GET','POST'])
def login():
    form = LoginForm()
    return render_template('login.html',form=form)

@app.route("/information", methods=['GET','POST'])
def information():
    content=request.form.get("content")
    print("content",content)
    return  content

@app.route('/read', methods=['GET','POST'])
def readtxt():
    with open("raw.txt",encoding="utf-8") as fr:
        lines=fr.readlines()
        for line in lines:
            print(line)
    return "读取成功"

@app.before_first_request
def before_first_request():
 session['click'] = 0


if __name__=="__main__":
    app.run()