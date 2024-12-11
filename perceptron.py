from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Example Dataset: OR gate
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 1]  # OR gate outputs

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Initialize Perceptron
model = Perceptron(max_iter=1000, tol=1e-3, random_state=42)

# Train the model
model.fit(X_train, y_train) 

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Test the model with all inputs
for inputs in X:
    prediction = model.predict([inputs])
    print(f"Input: {inputs}, Prediction: {prediction[0]}")
