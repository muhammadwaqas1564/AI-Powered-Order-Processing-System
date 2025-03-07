from flask import Flask, request, jsonify, render_template
from order_processor import OrderProcessor  # Import your OrderProcessor class here

app = Flask(__name__)
product_csv_path = "product.csv"  # Path to your product CSV
order_processor = OrderProcessor(product_csv_path)

@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    response = order_processor.handle_order(user_message)
    return jsonify({'reply': response})

if __name__ == "__main__":
    app.run(debug=True)
