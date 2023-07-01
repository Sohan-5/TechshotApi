import json
from flask import Flask, jsonify,request
import content_based

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Techshot Api"

@app.route('/recommend',methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            if not request.data:
                return jsonify({"error": "No input data"}),400
            data = json.loads(request.data)
            if not data.get("news_id"):
                return jsonify({"error": "No news_id in request data"}),400
            input_data = int(data["news_id"])
            output=content_based.recommend_articles(input_data,11)
            return jsonify(output),200
        except Exception as e:
            return jsonify({"error": str(e)}),400
    else:
        return "Not a valid method"

if __name__ == '__main__':
    app.run(debug=True,port=8001)
