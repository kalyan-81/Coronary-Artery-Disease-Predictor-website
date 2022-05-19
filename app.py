from flask import Flask, redirect, url_for, request, render_template
#from model import model_predict
import os
from werkzeug.utils import secure_filename
import requests
import cv2,numpy as np
import base64
import tensorflow.keras.models as tfm
app = Flask(__name__, static_url_path='/D:/m.tech/books/main project/New folder/CAD/basicfront-20220519T064649Z-001/basicfront/')


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('Frontpage.html')

@app.route('/home', methods=['GET'])
def home():
    # Main page
    return render_template('Frontpage.html')

@app.route('/single', methods=['GET'])
def single():
    # Main page
    return render_template('single.html')

@app.route('/twelve', methods=['GET'])
def twelve():
 # Main page
    return render_template('twelve.html')

@app.route('/about_us', methods=['GET'])
def about_us():
 # Main page
    return render_template('about_us.html')

@app.route('/contact_us', methods=['GET'])
def contact_us():
 # Main page
    return render_template('contact_us.html')




@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        #f = request.files['file']
        f1 = request.files['file']
        # f2 = request.form.get('scale')
        # f3 = request.form.get('image_type')
        
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f1.filename))
        f1.save(file_path)
        imggg=cv2.imread(file_path)
        img=cv2.resize(imggg,(150,150))
        _,_,rr=cv2.split(img)
        rr[rr>180]=255
        rr[rr<180]=0
        img=rr
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        # img=cv2.cvt
        filename=file_path+"OUT.jpg"
        cv2.imwrite(filename,img)
        img=img/255
        img=np.reshape(img,(1,150,150,3))
        # cv2.imshow("Hell",imggg)
        # cv2.waitKey(0)


        # scale = f2
        
        # with open(url_for('static', filename='scale.txt'),"w") as f_scale:
        #     f_scale.write(scale)
        #     f_scale.write(f3)
            
        # img_type = 1 if f3=="camera" else 0
        
        # if img_type==1:
        #     yml_file_path = os.path.join(basepath, 'options/df2k/test_df2k.yml')
        # else:
        #     yml_file_path = os.path.join(basepath, 'options/dped/test_dped.yml')       
        weightFile=os.getcwd()+"\Temp.h5"
        output_path = os.path.join(basepath, 'static', secure_filename(f1.filename))
        print("\n\n\n\n\n\n\n\n\n",os.listdir(),"\n\n\n\n\n\n")
        model=tfm.load_model('./nine2.h5',compile=False)
        k=float(model.predict(img)[0])

        print("\n\n\n\n\n\n\n",k,"\n\n\n\n\n\n")
        # r = requests.post("http://127.0.0.1:8080/predictions/super_res",files={'data':open(file_path,'rb')})
        # print(r,"HEioowhd---------------------------")
        # imgdata = base64.b64decode(r.content)
        # with open(output_path, 'wb') as f:
        # 	f.write(imgdata)
        #r.content.save(output_path)
        if k>0.5:
            return render_template('positive.html',fillename=secure_filename(f1.filename)+"OUT.jpg")
        else:
            return render_template('Negative.html',fillename=secure_filename(f1.filename)+"OUT.jpg")

         #https://stackoverflow.com/questions/46785507/python-flask-display-image-on-a-html-page
    return 'OK'

@app.route('/predict1/', methods=['GET', 'POST'])
def predict1():
    if request.method == 'POST':
        # Get the file from post request
        #f = request.files['file']
        f1 = request.files['file']
        # f2 = request.form.get('scale')
        # f3 = request.form.get('image_type')
        
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f1.filename))
        f1.save(file_path)
        imggg=cv2.imread(file_path)
        img=cv2.resize(imggg,(128,128))
        _,_,rr=cv2.split(img)
        rr[rr>180]=255
        rr[rr<180]=0
        img=rr
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        # img=cv2.cvt
        filename=file_path+"OUT.jpg"
        cv2.imwrite(filename,img)
        img=img/255
        img=np.reshape(img,(1,128,128,3))
        cv2.imshow("Hell",imggg)
        cv2.waitKey(0)
        
        # img1= cv2.imread(file_path)
        # BLACK_MIN = np.array([0,0,0],np.uint8)
        # BLACK_MAX = np.array([255,255,160],np.uint8)
        # hsv_img = cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)
        # frame_threshed = cv2.inRange(hsv_img, BLACK_MIN, BLACK_MAX)
        # cv2.imwrite('threshed.jpg', frame_threshed)
        # image = Image.open('threshed.jpg')
        # inverted_image = PIL.ImageOps.invert(image)
        # inverted_image.save('invert.jpg')
        # img2=cv2.imread("invert.jpg")
        # kernel = np.ones((2,2), np.uint8)
        # dilate = cv2.morphologyEx(img2, cv2.MORPH_DILATE, kernel)
        # fullpath = os.path.join(outPath, image_path)
        # cv2.imwrite(fullpath,dilate)

        # scale = f2
        
        # with open(url_for('static', filename='scale.txt'),"w") as f_scale:
        #     f_scale.write(scale)
        #     f_scale.write(f3)
            
        # img_type = 1 if f3=="camera" else 0
        
        # if img_type==1:
        #     yml_file_path = os.path.join(basepath, 'options/df2k/test_df2k.yml')
        # else:
        #     yml_file_path = os.path.join(basepath, 'options/dped/test_dped.yml')       
        weightFile=os.getcwd()+"\Temp.h5"
        output_path = os.path.join(basepath, 'static', secure_filename(f1.filename))
        print("\n\n\n\n\n\n\n\n\n",os.listdir(),"\n\n\n\n\n\n")
        model=tfm.load_model('./CADModel9797.h5',compile=False)
        k=float(model.predict(img)[0])

        print("\n\n\n\n\n\n\n",k,"\n\n\n\n\n\n")
        # r = requests.post("http://127.0.0.1:8080/predictions/super_res",files={'data':open(file_path,'rb')})
        # print(r,"HEioowhd---------------------------")
        # imgdata = base64.b64decode(r.content)
        # with open(output_path, 'wb') as f:
        # 	f.write(imgdata)
        #r.content.save(output_path)
        if k<0.5:
            return render_template('positive.html',fillename=secure_filename(f1.filename)+"OUT.jpg")
        else:
            return render_template('Negative.html',fillename=secure_filename(f1.filename)+"OUT.jpg")

         #https://stackoverflow.com/questions/46785507/python-flask-display-image-on-a-html-page
    return 'OK'

if __name__ == '__main__':
    app.run(debug=True)