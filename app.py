from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
from pathlib import Path


 
from fastai import *
from fastai.vision import * 

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

 
app = Flask(__name__)



path = Path("path")
classes = ['africangrey','budgerigar','cockatiel', 'cockatoo', 'conure','lovebird','macaw']
data2 = ImageDataBunch.single_from_classes(path, classes, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
learn = create_cnn(data2, models.resnet34)
learn.load('stage-1')




def model_predict(img_path): 
   
    img = open_image(img_path)
    pred_class,pred_idx,outputs = learn.predict(img)
	
    return str(pred_class)
    




@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST': 
        f = request.files['file']
 
        basepath = os.path.dirname(__file__)
        file_path = os.path.join( basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
 
        preds = model_predict(file_path)
        return preds
    return None


if __name__ == '__main__':
    
    app.run()


