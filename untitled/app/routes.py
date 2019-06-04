from app import app
from flask import render_template
from app.forms import  LoginForm
from app.Train import classfytrain
from flask import request


@app.route("/login", methods=['GET','POST'])
def login():
    form = LoginForm()
    return render_template('login.html',form=form)


@app.route("/information", methods=['GET','POST'])
def information():
    content=request.form.get("content")
    print("content",content)
    category=classfytrain(content)
    return category
