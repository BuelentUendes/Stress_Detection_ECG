import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Method 1: With DMatrix
start_time = time.time()

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'multi:softmax',
    'num_class': 3,
    'max_depth': 3,
    'learning_rate': 0.1
}

model_dmatrix = xgb.train(params, dtrain)
predictions_dmatrix = model_dmatrix.predict(dtest)
accuracy_dmatrix = accuracy_score(y_test, predictions_dmatrix)

dmatrix_time = time.time() - start_time

# Method 2: Without DMatrix (using sklearn API)
start_time = time.time()

model_simple = xgb.XGBClassifier(
    objective='multi:softmax',
    n_estimators=10,
    max_depth=3,
    learning_rate=0.1
)

model_simple.fit(X_train, y_train)
predictions_simple = model_simple.predict(X_test)
accuracy_simple = accuracy_score(y_test, predictions_simple)

simple_time = time.time() - start_time

# Print results
print(f'DMatrix Version:')
print(f'Accuracy: {accuracy_dmatrix:.2f}')
print(f'Time taken: {dmatrix_time:.4f} seconds\n')

print(f'Simple Version:')
print(f'Accuracy: {accuracy_simple:.2f}')
print(f'Time taken: {simple_time:.4f} seconds')
