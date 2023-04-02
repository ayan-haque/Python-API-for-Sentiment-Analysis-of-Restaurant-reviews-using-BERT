import sys
sys.path.append("<your_custom_path>/Sentiment-Analysis-of-Restaurant-Reviews")

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
        data = request.get_json()  # data = {"input_data": "It was delicious"}
        print(data) # Sample print
        results = Restaurant_Reviews_Obj.main(data)
        print(results)
    return jsonify(results)

if __name__ == '__main__':
    app.debug = True
    # serve(app, host = '0.0.0.0', port = 8080)
    app.run(threaded=True, host="127.0.0.1", port=5000) #runs the application on that local particular development server
