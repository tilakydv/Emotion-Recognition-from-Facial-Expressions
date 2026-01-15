import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_DIR = r"C:\Users\tilak\Downloads\Emotion_Recognition\Dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
IMG_SIZE = (48, 48)
BATCH_SIZE = 32
EPOCHS = 3
COLOR_MODE = "grayscale"
SAVE_DIR = "models"

os.makedirs(SAVE_DIR, exist_ok=True)

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)

if not os.path.isdir(TRAIN_DIR):
    raise FileNotFoundError(f"Train directory not found: {TRAIN_DIR}")
if not os.path.isdir(TEST_DIR):
    raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")

print("\nLoading training data...")
train_flow = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    color_mode=COLOR_MODE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

print("\nLoading testing data...")
test_flow = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    color_mode=COLOR_MODE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

num_classes = len(train_flow.class_indices)
print("\nDetected Classes:", train_flow.class_indices)
print("Number of Classes:", num_classes)

with open(os.path.join(SAVE_DIR, "class_indices.json"), "w") as fh:
    json.dump(train_flow.class_indices, fh, indent=2)

input_shape = (IMG_SIZE[0], IMG_SIZE[1], 1 if COLOR_MODE == "grayscale" else 3)

def build_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=input_shape, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

model = build_model(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\nModel Summary:")
model.summary()

checkpoint_path = os.path.join(SAVE_DIR, "best_model.h5")
checkpoint_cb = callbacks.ModelCheckpoint(
    checkpoint_path,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

earlystop_cb = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reduce_lr_cb = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    verbose=1
)

print("\nTraining started...\n")
history = model.fit(
    train_flow,
    validation_data=test_flow,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb, earlystop_cb, reduce_lr_cb]
)

final_model_path = os.path.join(SAVE_DIR, "final_model.h5")
model.save(final_model_path)
print(f"\nFinal model saved at: {final_model_path}")

if os.path.exists(checkpoint_path):
    try:
        print(f"\nLoading best model from checkpoint: {checkpoint_path}")
        model = tf.keras.models.load_model(checkpoint_path)
    except Exception as e:
        print(f"Could not load checkpoint due to: {e}. Using current model instead.")

def plot_history(history):
    if history is None:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(history.history.get("accuracy", []), label="Train Accuracy")
    plt.plot(history.history.get("val_accuracy", []), label="Val Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, "accuracy.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history.get("loss", []), label="Train Loss")
    plt.plot(history.history.get("val_loss", []), label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, "loss.png"))
    plt.close()

plot_history(history)

print("\nEvaluating model on test dataset...")
test_loss, test_acc = model.evaluate(test_flow)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")

print("\nGenerating predictions for confusion matrix...")
test_flow.reset()
y_true = test_flow.classes
y_pred_probs = model.predict(test_flow, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
labels = list(test_flow.class_indices.keys())
cm = confusion_matrix(y_true, y_pred)

def plot_confusion(cm, labels, name):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(os.path.join(SAVE_DIR, name))
    plt.close()

plot_confusion(cm, labels, "confusion_matrix.png")
print("\nTraining complete! All files saved in 'models/' folder.")

def show_predictions(model, data_dir, num_images=2, batch_size=8):
    preview_flow = test_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        color_mode=COLOR_MODE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True
    )
    class_indices = preview_flow.class_indices
    idx_to_class = {v: k for k, v in class_indices.items()}
    x_batch, y_batch = next(preview_flow)

    num_images = min(num_images, x_batch.shape[0])

    for i in range(num_images):
        img = x_batch[i]
        true_idx = np.argmax(y_batch[i])
        true_label = idx_to_class[true_idx]

        pred_prob = model.predict(img.reshape(1, *img.shape))[0]
        pred_idx = np.argmax(pred_prob)
        pred_label = idx_to_class[pred_idx]
        pred_conf = pred_prob[pred_idx]

        plt.figure(figsize=(5, 5))
        if COLOR_MODE == "grayscale":
            disp = img.squeeze()
            plt.imshow(disp, cmap="gray")
        else:
            disp = np.clip(img, 0, 1)
            plt.imshow(disp)
        plt.title(f"True: {true_label}\nPred: {pred_label} ({pred_conf*100:.1f}%)")
        plt.axis("off")

        save_path = os.path.join(
            SAVE_DIR,
            f"pred_img_{i+1}_{true_label}_pred_{pred_label}.png"
        )
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"Saved {save_path}")

    print(f"\nSaved sample prediction images in {SAVE_DIR}")

print("\nShowing sample predictions...")
show_predictions(model, TEST_DIR, num_images=2, batch_size=BATCH_SIZE)

