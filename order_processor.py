import spacy
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from typing import Dict, List, Tuple, Optional
import inflect
import re
import logging

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize inflect engine for handling singular/plural forms
p = inflect.engine()

# Number word mapping
NUMBER_WORDS = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
    'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
    'nineteen': 19, 'twenty': 20
}

# Set up logging for error handling and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OrderProcessor:
    def __init__(self, product_csv: str):
        # Initialize product data and processing logic
        self.product_db = self._load_product_data(product_csv)
        self.product_forms = self._create_product_forms()
        self.intent_classifier = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=50000)),
            ('clf', LinearSVC())
        ])
        self._train_intent_classifier()

    def _load_product_data(self, product_csv: str) -> Dict[str, Dict[str, any]]:
        df = pd.read_csv(product_csv)
        return {row['name'].lower(): {'price': row['price'], 'stock': row['stock']} for _, row in df.iterrows()}

    def _word_to_number(self, word: str) -> Optional[int]:
        word = word.lower()
        try:
            return int(word)
        except ValueError:
            return NUMBER_WORDS.get(word)

    def _create_product_forms(self) -> Dict[str, str]:
        product_forms = {}
        for product in self.product_db.keys():
            product_forms[product] = product
            product_forms[product.lower()] = product
            if ' ' not in product:
                plural = p.plural(product)
                product_forms[plural] = product
            if 'iphone' in product:
                parts = product.split()
                if len(parts) == 2:
                    base_name, model = parts
                    variations = [
                        f"iphone{model}", f"iphone {model}", f"iPhone{model}", f"iPhone {model}",
                        f"iphone-{model}", f"iPhone-{model}"
                    ]
                    for variation in variations:
                        product_forms[variation.lower()] = product
            if 'macbook' in product:
                parts = product.split()
                if len(parts) == 2:
                    base_name, model = parts
                    variations = [
                        f"macbook{model}", f"macbook-{model}", f"MacBook{model.capitalize()}",
                        f"MacBook {model.capitalize()}", f"MacBook-{model.capitalize()}"
                    ]
                    for variation in variations:
                        product_forms[variation.lower()] = product
        return product_forms

    def _train_intent_classifier(self):
        X_train = [
            "I want to buy a laptop",
            "Please order two phones",
            "Add three tablets to cart",
            "I’d like to buy a camera.",
            "Order me a new monitor.",
            "Can you get me a wireless mouse?",
            "I'd like to order a keyboard.",
            "Please add a smartwatch to my order.",
            "Can you help me buy a gaming console?",
            "I need a printer, please.",
            "I want to buy an iPhone 12",
            "Order MacBook Pro",
            "Purchase headphones",
            "What's the weather like?",
            "Tell me a joke",
            "What is the latest smartphone?",
            "Tell me a story.",
            "How do I cook spaghetti?",
            "What time is it?",
            "Give me some travel tips.",
            "What's the news today?",
            "Tell me about your services.",
            "Can you recommend a good movie?"
        ]
        y_train = [
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]
        self.intent_classifier.fit(X_train, y_train)

    def _find_product_in_text(self, text: str) -> List[Tuple[str, int, int]]:
        text = text.lower()
        found_products = []
        sorted_product_forms = sorted(self.product_forms.keys(), key=len, reverse=True)
        for product_form in sorted_product_forms:
            for match in re.finditer(r'\b' + re.escape(product_form) + r'\b', text):
                found_products.append((self.product_forms[product_form], match.start(), match.end()))
        found_products.sort(key=lambda x: x[1])
        filtered_products = []
        last_end = -1
        for product, start, end in found_products:
            if start >= last_end:
                filtered_products.append((product, start, end))
                last_end = end
        return filtered_products

    def preprocess_text(self, text: str) -> str:
        doc = nlp(text.lower().strip())
        tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
        return " ".join(tokens)

    def classify_intent(self, text: str) -> bool:
        processed_text = self.preprocess_text(text)
        prediction = self.intent_classifier.predict([processed_text])[0]
        return bool(prediction)

    def extract_entities(self, text: str) -> List[Tuple[str, int]]:
        text = text.lower()
        products = []
        quantities = {}
        number_pattern = r'\b(\d+|' + '|'.join(NUMBER_WORDS.keys()) + r')\b'
        for match in re.finditer(number_pattern, text):
            number = self._word_to_number(match.group(1))
            if number is not None:
                quantities[match.start()] = number
        found_products = self._find_product_in_text(text)
        for product, start, end in found_products:
            nearest_quantity = 1
            nearest_distance = float('inf')
            for qty_pos, qty in quantities.items():
                distance = start - qty_pos
                if 0 < distance < nearest_distance:
                    nearest_quantity = qty
                    nearest_distance = distance
            products.append((product, nearest_quantity))
        return products

    def validate_order(self, products: List[Tuple[str, int]]) -> Tuple[bool, str]:
        for product, quantity in products:
            if product not in self.product_db:
                return False, f"Product {product} not found."
            if self.product_db[product]['stock'] < quantity:
                return False, f"Insufficient stock for {product}."
        return True, "Order validated successfully."

    def calculate_total(self, products: List[Tuple[str, int]]) -> float:
        return sum(self.product_db[product]['price'] * quantity for product, quantity in products)

    def process_payment(self, total: float) -> bool:
        return True if total > 0 else False

    def update_inventory(self, products: List[Tuple[str, int]]) -> None:
        for product, quantity in products:
            self.product_db[product]['stock'] -= quantity

    def check_availability_or_price(self, text: str) -> str:
        """Check the availability or price of a product based on user input."""
        processed_text = self.preprocess_text(text)
        found_products = self._find_product_in_text(processed_text)

        if not found_products:
            return "No products found in the request."

        responses = []
        for product, start, end in found_products:
            if product in self.product_db:
                price = self.product_db[product]['price']
                stock = self.product_db[product]['stock']
                if stock > 0:
                    responses.append(f"{product.capitalize()} is available for ${price:.2f}.")
                else:
                    responses.append(f"{product.capitalize()} is out of stock.")
            else:
                responses.append(f"{product.capitalize()} not found in the database.")

        return " ".join(responses)

    def list_available_products(self) -> str:
        if not self.product_db:
            return "No products available at the moment."

        available_products = ', '.join(self.product_db.keys())
        return f"Available products: {available_products}."

    def handle_greeting(self, text: str) -> Optional[str]:
        greetings = ["hi", "hello", "hey", "hyy", "helo"]
        if any(greet in text.lower() for greet in greetings):
            return "Hey! How can I help you today? Would you like to make a purchase today?"
        return None

    def handle_confirmation(self, text: str) -> str:
        if "yes" in text.lower():
            return self.list_available_products()
        return "Alright, let me know if you need anything else!"

    def collect_user_details(self) -> Dict[str, str]:
        """Collect user details for order processing."""
        details = {}
        
        details['name'] = input("Please enter your name: ")
        details['address'] = input("Please enter your address: ")
        details['credit_card'] = input("Please enter your credit card details: ")
        
        # Confirm the collected details
        print("\nPlease confirm your details:")
        for key, value in details.items():
            print(f"{key.capitalize()}: {value}")
        
        confirm = input("Are these details correct? (yes/no): ").strip().lower()
        if confirm != 'yes':
            print("Let's try again.")
            return self.collect_user_details()  # Restart collection if details are incorrect
        
        return details

    def handle_order(self, text: str) -> str:
        """Process customer order from input text."""
        greeting_response = self.handle_greeting(text)
        if greeting_response:
            print(greeting_response)
            return "Please respond with 'yes' to view available products."

        confirmation_response = self.handle_confirmation(text)
        if confirmation_response != "Alright, let me know if you need anything else!":
            return confirmation_response

        if "price" in text.lower() or "available" in text.lower() or "stock" in text.lower():
            return self.check_availability_or_price(text)

        if "what products do you have" in text or "available products" in text or "products" in text:
            return self.list_available_products()

        if not self.classify_intent(text):
            return "No order intent detected. Please make sure you're asking for an order or product-related inquiry."

        # Collect user details before processing the order
        user_details = self.collect_user_details()
        print(f"User Details Collected: {user_details}")

        products = self.extract_entities(text)

        if not products:
            return "No products found in the order."

        valid, message = self.validate_order(products)
        if not valid:
            return message

        total = self.calculate_total(products)
        if not self.process_payment(total):
            return "Payment failed."

        self.update_inventory(products)
        return f"Order processed successfully. Total: ${total:.2f}. Thank you, {user_details['name']}! Your order will be shipped to {user_details['address']}."

if __name__ == "__main__":
    product_csv_path = "product.csv"  # Update the path as needed

    # Initialize the OrderProcessor with the product CSV
    order_processor = OrderProcessor(product_csv_path)

    while True:
        # Prompt the user for input
        user_input = input("Please enter your order (or type 'exit' to quit): ")

        # Exit the loop if the user types 'exit'
        if user_input.lower() == 'exit':
            print("Exiting the order system. Thank you!")
            break

        # Process the user's order and print the result
        result = order_processor.handle_order(user_input)
        print(result)



