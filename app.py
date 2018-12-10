from flask import Flask, jsonify, render_template, request
import os
import sys
import base64
from binascii import a2b_base64
from mimetypes import guess_extension
from urllib.request import urlretrieve
from urllib.parse import unquote
from server import predict



app = Flask(__name__, static_folder="./static/dist", template_folder="./static")

'''
    Method to generate new face
    @return [String] success message
'''
@app.route('/_generate')
def _generate():
    startUploading = request.args.get('startUploading', 0, type=bool)
    type = request.args.get('type')
    input = request.args.get('input')
    mask = request.args.get('mask')
    x1 = request.args.get('mask1')
    y1 = request.args.get('mask2')
    x2 = request.args.get('mask3')
    y2 = request.args.get('mask4')

    maskname = saveImage(mask, 'mask', 'png');

    if (type == 'jpg' or type == 'png'  ):
        filename = saveImage(input, 'from_upload', type);
    elif (type == 'url'):
        filename = input



    message = 'Image successfully updated'
    # newFaceSrcUrl = predict.generateNewFace(filename, type, x1, y1, x2, y2)
    newFaceSrcUrl = predict.generateNewFace(filename, type, maskname)
    return jsonify(result=newFaceSrcUrl, msg=message)

'''
    Method to render main entry html page for Flask
'''
@app.route('/')
def index():
    return render_template('index.html')

'''
    Method to save dataUrl to local image
'''
def saveImage(dataURLVar, name, type):
    fname = 'server/' + name + '.' + type
    fname2 = 'static/dist/' + name + '.' + type
    result = 'dist/' + name + '.' + type
    # imgData = base64.b64decode(dataURLVar)
    dataURL = unquote(dataURLVar)
    imgdata=base64.b64decode(dataURL)
    file=open(fname,'wb')
    file.write(imgdata)
    file.close()
    file2=open(fname2,'wb')
    file2.write(imgdata)
    file2.close()

    return fname

if __name__ == '__main__':
    app.run(debug=False)
