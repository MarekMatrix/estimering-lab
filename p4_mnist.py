# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
#   "matplotlib",
#   "scikit-learn",
#   "scipy",
# ]
# ///

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.spatial.distance import cdist


def load_mnist():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    X = mnist.data / 255.0
    y = mnist.target.astype(int)
    return X[:60000], y[:60000], X[60000:], y[60000:]


def evaluate(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    error_rate = 1 - accuracy_score(y_true, y_pred)
    print(f"{name} — Error rate: {error_rate:.4f} ({error_rate*100:.2f}%)")
    return cm, error_rate


def nn_classify(templates, template_labels, test, chunk_size=1000):
    predictions = np.empty(len(test), dtype=int)
    for i in range(0, len(test), chunk_size):
        chunk = test[i:i + chunk_size]
        dists = cdist(chunk, templates, metric='euclidean')
        predictions[i:i + chunk_size] = template_labels[np.argmin(dists, axis=1)]
    return predictions


def plot_images(X_test, y_test, y_pred, misclassified=True, title=''):
    if misclassified:
        mask = y_pred != y_test
    else:
        mask = y_pred == y_test
    indices = np.where(mask)[0][:9]
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    for ax, idx in zip(axes.flat, indices):
        ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
        ax.set_title(f'T:{y_test[idx]} P:{y_pred[idx]}', fontsize=8)
        ax.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def cluster_templates(X_train, y_train, M=64):
    all_templates = []
    all_labels = []
    for c in range(10):
        Xc = X_train[y_train == c]
        km = KMeans(n_clusters=M, random_state=42, n_init=10)
        km.fit(Xc)
        all_templates.append(km.cluster_centers_)
        all_labels.extend([c] * M)
        print(f"  Class {c} clustered")
    return np.vstack(all_templates), np.array(all_labels)


print("Loading MNIST...")
X_train, y_train, X_test, y_test = load_mnist()
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

print("\n--- Part 1: Full NN (60k templates) ---")
t0 = time.time()
pred_nn_full = nn_classify(X_train, y_train, X_test, chunk_size=1000)
t1_full = time.time() - t0
print(f"Classification time: {t1_full:.1f}s")
cm1, err1 = evaluate(y_test, pred_nn_full, "NN (full)")
print("Confusion matrix:")
print(cm1)

plot_images(X_test, y_test, pred_nn_full, misclassified=True, title='Part 1 — Misclassified (NN full)')
plot_images(X_test, y_test, pred_nn_full, misclassified=False, title='Part 1 — Correctly Classified (NN full)')

print("\n--- Part 2a: K-Means clustering (M=64 per class) ---")
t0 = time.time()
templates_km, labels_km = cluster_templates(X_train, y_train, M=64)
t_cluster = time.time() - t0
print(f"Clustering time: {t_cluster:.1f}s | Templates shape: {templates_km.shape}")

print("\n--- Part 2b: NN on 64 templates/class ---")
t0 = time.time()
pred_nn_km = nn_classify(templates_km, labels_km, X_test, chunk_size=1000)
t2 = time.time() - t0
print(f"Classification time: {t2:.1f}s")
cm2, err2 = evaluate(y_test, pred_nn_km, "NN (64 templates/class)")
print("Confusion matrix:")
print(cm2)

print("\n--- Part 2c: K-NN (K=7) on 64 templates/class ---")
t0 = time.time()
knn = KNeighborsClassifier(n_neighbors=7, metric='euclidean')
knn.fit(templates_km, labels_km)
pred_knn = knn.predict(X_test)
t3 = time.time() - t0
print(f"Classification time: {t3:.1f}s")
cm3, err3 = evaluate(y_test, pred_knn, "K-NN K=7 (64 templates/class)")
print("Confusion matrix:")
print(cm3)

print("\n--- Summary ---")
print(f"{'System':<30} {'Error Rate':>12} {'Time (s)':>10}")
print(f"{'NN (full 60k templates)':<30} {err1:>12.4f} {t1_full:>10.1f}")
print(f"{'NN (64 templates/class)':<30} {err2:>12.4f} {t2:>10.1f}")
print(f"{'K-NN K=7 (64 templates/class)':<30} {err3:>12.4f} {t3:>10.1f}")
