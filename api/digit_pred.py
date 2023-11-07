from flask import Flask, request

app = Flask(__name__)

@app.route('/model', methods=['POST'])
def pred_model():
    js = request.get_json()
    img1 = js['img1']
    img2 = js['img2']
    #Assuming this is the path of our best trained model
    model = load('./models/Candidate_Model_tree_(10,).joblib')
    pred_img1 = model.predict(img1)
    pred_img2 = model.predict(img2)
    if(pred_img1 == pred_img2):
        return True
    else:
        return False