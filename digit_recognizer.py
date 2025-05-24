
#Libraries
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
# Load the digits dataset
digits = load_digits()
df = pd.DataFrame(data=digits.data)
df['target'] = digits.target

# Split the dataset into training and testing sets
X = df.drop('target', axis=1) # Features
y = df['target'] # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Create a KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
accuracies = []
# Test different values of k
for k in range(1,11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = (accuracy_score(y_test, y_pred))
    accuracies.append(accuracy)
    print(f"Accuracy for k={k}: {accuracy * 100:.2f}%")
# Visualize some of the digits
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test.iloc[i].values.reshape(8, 8), cmap='gray')
    plt.title(f'Predicted: {y_pred[i]}')
    plt.axis('off')
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(range(1, 11), accuracies, marker='o' , linestyle='-')
plt.title('Accuracy score')
plt.xlabel('K Value')
plt.ylabel('Accuracy score')
plt.grid(True)
plt.show()
#Print the index of wrong predictions
for i in range(len(y_pred)):
    if y_pred[i] != y_test.iloc[i]:
        print(f"Index: {i} , True {y_test.iloc[i]}, Predicted: {y_pred[i]},")


