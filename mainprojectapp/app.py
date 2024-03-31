from flask import Flask, render_template,session,send_file,request
import torch
import pathlib
import io
import pandas 
import numpy as np
import cv2
import os
import PIL.Image as Image
pathlib.PosixPath = pathlib.WindowsPath

app = Flask(__name__,static_folder='static')


#global imgname

@app.route('/',methods=['GET','POST'])
def index():
    output = None
    #value = False
    if request.method=='POST':
        if request.files['pcbandwood']:
            file = request.files['pcbandwood']
            global imageName
            img =  Image.open(io.BytesIO(file.read()))
            pcb_wood_confidence = request.form['pcbandwoodconfidence']
            pcb_wood_threshold = request.form['pcbandwoodthreshold']
            model = torch.hub.load(r'models/pcbandwoodmodel','custom',path=r'models/pcbandwoodmodel/runs/train/exp/weights/best.pt',source='local',force_reload=True)
            model.conf = float(pcb_wood_confidence)
            model.iou = float(pcb_wood_threshold)
            result = model(img)
            #result.print()
            output = result.pandas().xyxy[0].name
            #output = result.pandas().xyxy[0].to_json(orient="records")
            value = True
            result.save(save_dir='static',exist_ok=True)
            return render_template('homepage.html',front = output,validate = value)
        if request.files['pcbcomponent']:
            file = request.files['pcbcomponent']
            img =  Image.open(io.BytesIO(file.read()))
            pcb_component_confidence = request.form['pcbcomponentconfidence']
            pcb_component_threshold = request.form['pcbcomponentthreshold']
            model = torch.hub.load(r'models/pcbcomponentmodel','custom',path=r'models/pcbcomponentmodel/runs/train/exp/weights/best.pt',source='local',force_reload=True)
            model.conf = float(pcb_component_confidence)
            model.iou = float(pcb_component_threshold)
            result = model(img)
            #result.print()
            output = result.pandas().xyxy[0].name
            #output = result.pandas().xyxy[0].to_json(orient="records")
            value = True
            #filepath = os.path.join('runs/detect/',file.filename)
            result.save(save_dir='static',exist_ok=True)
            return render_template('homepage.html',front = output,validate = value)
        if request.files['pcbfrontdefect']:
            file = request.files['pcbfrontdefect']
            img =  Image.open(io.BytesIO(file.read()))
            pcb_front_confidence = request.form['pcbfrontconfidence']
            pcb_front_threshold = request.form['pcbfrontthreshold']
            model = torch.hub.load(r'models/pcbfrontmodel','custom',path=r'models/pcbfrontmodel/runs/train/exp/weights/best.pt',source='local',force_reload=True)
            model.conf = float(pcb_front_confidence)
            model.iou = float(pcb_front_threshold)
            result = model(img)
            #result.print()
            output = result.pandas().xyxy[0].name
            #output = result.pandas().xyxy[0].to_json(orient="records")
            value = True
            #filepath = os.path.join('runs/detect/',file.filename)
            result.save(save_dir='static',exist_ok=True)
            return render_template('homepage.html',front = output,validate = value)
    return render_template('homepage.html')



if __name__ == "__main__":
    app.run()