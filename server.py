import sys
sys.path.append("<your_custom_path>/Python-API-for-Sentiment-Analysis-of-Restaurant-reviews-using-BERT")
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

from main_file import Restaurant_Reviews
Restaurant_Reviews_Obj = Restaurant_Reviews()

@app.route('/')
def Test_check():
    """
    Test route to check the port
    """
    return 'Hello World'

@app.route('/validate', methods = ['POST'])
def validate_review():
    if request.method == 'POST':
        data = request.get_json()
        print(data) # Sample print
        results = Restaurant_Reviews_Obj.main(data)
        print(results)
    return jsonify(results)

if __name__ == '__main__':
    app.debug = True
    app.run(threaded=True, host="127.0.0.1", port=5000)
