from flask_wtf import  FlaskForm
from  wtforms import  StringField,SubmitField
from wtforms.validators import DataRequired

class LoginForm(FlaskForm):
    content = StringField("content",validators=[DataRequired()])
    category = StringField("category",validators=[DataRequired()])
    submit = SubmitField('提交')