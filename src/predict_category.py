import joblib
import pandas as pd

# Load the saved model
model = joblib.load("model/product_classifier.pkl")

# Prediction function
def predict_category(product_title):
    try:
        prediction = model.predict([product_title])[0]
        return prediction
    except Exception as e:
        return f"Error: {str(e)}"



print("Enter a product name for classification.")
print("Write 'exit' to leave.")


while True:
    # User input
    user_input = input("\nEnter a product name: ").strip()
    
    # Exiting the loop
    if user_input.lower() in ['exit']:
        print("Bye!")
        break
    
    # If the input is empty
    if not user_input:
        print("Enter a valid product:")
        continue
    
    # Prediction
    print("Classifying the product...")
    prediction = predict_category(user_input)
    
    # Result
    print(f"Category: {prediction}")
    print("-" * 40)