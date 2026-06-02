"""
Handwritten Character Recognition — CodeAlpha Machine Learning Internship
Task 3: Identify handwritten digits using CNN on MNIST dataset.
Author: Allen Stivanson Christian
Student ID: CA/DF1/110227
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")
os_environ = __import__("os").environ
os_environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ── 1. Load & Preprocess Data ─────────────────────────────────────────────────
def load_data():
    print("\n [1/6] Loading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalize to [0, 1]
    X_train = X_train.astype("float32") / 255.0
    X_test  = X_test.astype("float32")  / 255.0

    # Reshape for CNN: (samples, height, width, channels)
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test  = X_test.reshape(-1, 28, 28, 1)

    # One-hot encode labels
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat  = to_categorical(y_test,  10)

    print(f"  ✔ Training samples : {X_train.shape[0]}")
    print(f"  ✔ Testing samples  : {X_test.shape[0]}")
    print(f"  ✔ Image shape      : {X_train.shape[1:]}")
    print(f"  ✔ Classes          : 0-9 (10 digits)")

    return X_train, X_test, y_train, y_test, y_train_cat, y_test_cat

# ── 2. EDA ────────────────────────────────────────────────────────────────────
def run_eda(X_train, y_train):
    print("\n [2/6] Running Exploratory Data Analysis...")

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("Handwritten Digit Recognition — EDA", fontsize=14, fontweight="bold")

    gs = gridspec.GridSpec(2, 3, figure=fig)

    # Sample digits grid
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.axis("off")
    ax1.set_title("Sample Digits from MNIST Dataset", fontsize=11, pad=10)

    inner_gs = gridspec.GridSpecFromSubplotSpec(4, 10, subplot_spec=gs[0, :2],
                                                 hspace=0.1, wspace=0.1)
    for digit in range(10):
        indices = np.where(y_train == digit)[0][:4]
        for row, idx in enumerate(indices):
            ax = fig.add_subplot(inner_gs[row, digit])
            ax.imshow(X_train[idx].reshape(28, 28), cmap="gray")
            ax.axis("off")
            if row == 0:
                ax.set_title(str(digit), fontsize=9)

    # Class distribution
    ax2 = fig.add_subplot(gs[0, 2])
    unique, counts = np.unique(y_train, return_counts=True)
    bars = ax2.bar(unique, counts, color=plt.cm.tab10(np.linspace(0, 1, 10)),
                   edgecolor="black", alpha=0.85)
    ax2.set_title("Class Distribution")
    ax2.set_xlabel("Digit")
    ax2.set_ylabel("Count")
    ax2.set_xticks(range(10))
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                str(count), ha="center", va="bottom", fontsize=7)

    # Pixel intensity distribution
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(X_train.flatten(), bins=50, color="#3498db", alpha=0.75, edgecolor="black")
    ax3.set_title("Pixel Intensity Distribution")
    ax3.set_xlabel("Pixel Value (normalized)")
    ax3.set_ylabel("Frequency")

    # Average digit images
    ax4 = fig.add_subplot(gs[1, 1])
    avg_images = np.array([X_train[y_train == d].mean(axis=0).reshape(28,28)
                           for d in range(10)])
    combined = np.concatenate([avg_images[i] for i in range(10)], axis=1)
    ax4.imshow(combined, cmap="hot")
    ax4.set_title("Average Image per Digit (0-9)")
    ax4.set_xticks([14 + 28*i for i in range(10)])
    ax4.set_xticklabels(range(10))
    ax4.set_yticks([])

    # Sample with labels
    ax5 = fig.add_subplot(gs[1, 2])
    random_idx = np.random.choice(len(X_train), 25, replace=False)
    grid = np.zeros((5*28, 5*28))
    labels_grid = []
    for i, idx in enumerate(random_idx):
        r, c = divmod(i, 5)
        grid[r*28:(r+1)*28, c*28:(c+1)*28] = X_train[idx].reshape(28,28)
        labels_grid.append(y_train[idx])
    ax5.imshow(grid, cmap="gray")
    ax5.set_title("Random 25 Samples")
    ax5.axis("off")

    plt.tight_layout()
    plt.savefig("eda_digits.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✔ EDA saved to 'eda_digits.png'")

# ── 3. Build CNN Model ────────────────────────────────────────────────────────
def build_model():
    model = keras.Sequential([
        # Block 1
        layers.Conv2D(32, (3,3), activation="relu", padding="same",
                      input_shape=(28,28,1), name="conv1"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), activation="relu", padding="same", name="conv2"),
        layers.MaxPooling2D((2,2), name="pool1"),
        layers.Dropout(0.25, name="drop1"),

        # Block 2
        layers.Conv2D(64, (3,3), activation="relu", padding="same", name="conv3"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation="relu", padding="same", name="conv4"),
        layers.MaxPooling2D((2,2), name="pool2"),
        layers.Dropout(0.25, name="drop2"),

        # Classifier
        layers.Flatten(name="flatten"),
        layers.Dense(256, activation="relu", name="dense1"),
        layers.BatchNormalization(),
        layers.Dropout(0.5, name="drop3"),
        layers.Dense(10, activation="softmax", name="output")
    ], name="HandwrittenCNN")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ── 4. Train Model ────────────────────────────────────────────────────────────
def train_model(model, X_train, y_train_cat, X_test, y_test_cat):
    print("\n [4/6] Training CNN Model...")

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=5,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=3, verbose=1, min_lr=1e-6)
    ]

    # Data augmentation
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    datagen.fit(X_train)

    history = model.fit(
        datagen.flow(X_train, y_train_cat, batch_size=128),
        epochs=20,
        validation_data=(X_test, y_test_cat),
        callbacks=callbacks,
        verbose=1
    )
    return history

# ── 5. Plot Training History ──────────────────────────────────────────────────
def plot_results(history, model, X_test, y_test, y_test_cat):
    print("\n [5/6] Generating result visualizations...")

    y_pred     = model.predict(X_test, verbose=0)
    y_pred_cls = np.argmax(y_pred, axis=1)

    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    fig.suptitle("Handwritten Digit Recognition — CNN Results", fontsize=14, fontweight="bold")

    # Training accuracy
    axes[0,0].plot(history.history["accuracy"], label="Train", color="#3498db", linewidth=2)
    axes[0,0].plot(history.history["val_accuracy"], label="Validation", color="#e74c3c", linewidth=2)
    axes[0,0].set_title("Model Accuracy")
    axes[0,0].set_xlabel("Epoch")
    axes[0,0].set_ylabel("Accuracy")
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # Training loss
    axes[0,1].plot(history.history["loss"], label="Train", color="#3498db", linewidth=2)
    axes[0,1].plot(history.history["val_loss"], label="Validation", color="#e74c3c", linewidth=2)
    axes[0,1].set_title("Model Loss")
    axes[0,1].set_xlabel("Epoch")
    axes[0,1].set_ylabel("Loss")
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_cls)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0,2],
                xticklabels=range(10), yticklabels=range(10))
    axes[0,2].set_title("Confusion Matrix")
    axes[0,2].set_xlabel("Predicted")
    axes[0,2].set_ylabel("Actual")

    # Correct predictions
    correct = np.where(y_pred_cls == y_test)[0]
    axes[1,0].set_title("Correctly Classified Samples")
    axes[1,0].axis("off")
    inner = gridspec.GridSpecFromSubplotSpec(3, 6,
                subplot_spec=axes[1,0].get_subplotspec(), hspace=0.3, wspace=0.1)
    for i in range(18):
        ax = fig.add_subplot(inner[i//6, i%6])
        ax.imshow(X_test[correct[i]].reshape(28,28), cmap="gray")
        ax.set_title(f"✓{y_test[correct[i]]}", fontsize=7, color="green")
        ax.axis("off")

    # Wrong predictions
    wrong = np.where(y_pred_cls != y_test)[0]
    axes[1,1].set_title("Misclassified Samples")
    axes[1,1].axis("off")
    inner2 = gridspec.GridSpecFromSubplotSpec(3, 6,
                subplot_spec=axes[1,1].get_subplotspec(), hspace=0.3, wspace=0.1)
    for i in range(min(18, len(wrong))):
        ax = fig.add_subplot(inner2[i//6, i%6])
        ax.imshow(X_test[wrong[i]].reshape(28,28), cmap="Reds")
        ax.set_title(f"P:{y_pred_cls[wrong[i]]} A:{y_test[wrong[i]]}", fontsize=6, color="red")
        ax.axis("off")

    # Per class accuracy
    class_acc = cm.diagonal() / cm.sum(axis=1)
    bars = axes[1,2].bar(range(10), class_acc,
                          color=plt.cm.RdYlGn(class_acc), edgecolor="black")
    axes[1,2].set_title("Per-Class Accuracy")
    axes[1,2].set_xlabel("Digit Class")
    axes[1,2].set_ylabel("Accuracy")
    axes[1,2].set_xticks(range(10))
    axes[1,2].set_ylim(0.9, 1.01)
    for bar, acc in zip(bars, class_acc):
        axes[1,2].text(bar.get_x() + bar.get_width()/2,
                       bar.get_height() + 0.001,
                       f"{acc:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig("cnn_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✔ Results saved to 'cnn_results.png'")
    return y_pred_cls

# ── 6. Main ───────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("   HANDWRITTEN CHARACTER RECOGNITION")
    print("   CodeAlpha ML Internship — Task 3")
    print("   Allen Stivanson Christian | CA/DF1/110227")
    print("="*60)

    # Load data
    X_train, X_test, y_train, y_test, y_train_cat, y_test_cat = load_data()

    # EDA
    run_eda(X_train, y_train)

    # Build model
    print("\n [3/6] Building CNN Architecture...")
    model = build_model()
    model.summary()

    # Train
    history = train_model(model, X_train, y_train_cat, X_test, y_test_cat)

    # Evaluate
    print("\n [5/6] Evaluating Model...")
    loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\n {'─'*50}")
    print(f"  🏆 Final Test Accuracy : {accuracy*100:.2f}%")
    print(f"  📉 Final Test Loss     : {loss:.4f}")
    print(f" {'─'*50}")

    y_pred_cls = plot_results(history, model, X_test, y_test, y_test_cat)

    # Classification report
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred_cls,
                                target_names=[str(i) for i in range(10)]))

    # Save model
    model.save("handwritten_cnn_model.keras")
    print("  ✔ Model saved to 'handwritten_cnn_model.keras'")

    print("\n" + "="*60)
    print("  ✅ Handwritten Recognition — Complete!")
    print("  📊 Check 'eda_digits.png' and 'cnn_results.png'")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
