from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.train_models import train_logistic, train_decision_tree, train_random_forest
from src.evaluate import evaluate_model
from sklearn.model_selection import train_test_split

# Load the dataset
df = load_data('data/tested.csv')

# Preprocess the data
X, y = preprocess_data(df)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
log_model = train_logistic(X_train, y_train)
dt_model = train_decision_tree(X_train, y_train)
rf_model = train_random_forest(X_train, y_train)

# Evaluate models
print("=== Logistic Regression ===")
evaluate_model(log_model, X_test, y_test)

print("\n=== Decision Tree ===")
evaluate_model(dt_model, X_test, y_test)

print("\n=== Random Forest ===")
evaluate_model(rf_model, X_test, y_test)
