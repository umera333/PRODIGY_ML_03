import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib
import matplotlib.pyplot as plt

# Configuration
data_dir = r'C:\Users\Ananya\OneDrive\Documents\dogs-vs-cats-svm\train\train'  # Full path to image folder
img_size = 64      # Resize images to 64x64
limit = 2000       # Total images to use (e.g. 1000 cats + 1000 dogs)
test_size = 0.2    # Test/train split
model_filename = "svm_dog_vs_cat_model.pkl"

def load_images(data_dir, img_size, limit):
    X, y = [], []
    cat_count = dog_count = 0
    print(f"Looking for images in: {data_dir}")
    files = os.listdir(data_dir)
    print("Files found in folder:")
    print(files[:10])  # Preview first 10 files

    for file in files:
        if not file.endswith('.jpg'):
            continue
        label = 0 if 'cat' in file.lower() else 1
        if (label == 0 and cat_count >= limit // 2) or (label == 1 and dog_count >= limit // 2):
            continue
        path = os.path.join(data_dir, file)
        img = cv2.imread(path)
        if img is None:
            continue  # Skip unreadable images
        img = cv2.resize(img, (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        X.append(img.flatten())
        y.append(label)
        if label == 0:
            cat_count += 1
        else:
            dog_count += 1

    print(f"Loaded {cat_count} cat and {dog_count} dog images.")
    return np.array(X), np.array(y)

def visualize_sample(X, y, img_size, count=5):
    plt.figure(figsize=(10, 2))
    for i in range(count):
        img = X[i].reshape((img_size, img_size))
        label = "Cat" if y[i] == 0 else "Dog"
        plt.subplot(1, count, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(label)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Load dataset
X, y = load_images(data_dir, img_size, limit)
print(f"Total images loaded: {len(X)}")

if len(np.unique(y)) < 2:
    print("ERROR: Dataset must contain both cats and dogs.")
    exit()

# Optional: visualize sample images
# visualize_sample(X, y, img_size)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

# Train the SVM model
print("Training the SVM model...")
svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train, y_train)

# Save the model
joblib.dump(svm, model_filename)
print(f"Model saved to {model_filename}")

# Evaluate
y_pred = svm.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))