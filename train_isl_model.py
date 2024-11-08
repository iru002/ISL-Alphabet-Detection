import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Define image size
IMG_WIDTH, IMG_HEIGHT = 64, 64

# Define data directory
data_dir = 'isl/Indian'

# Function to load dataset
def load_data(data_dir, limit_per_class=None):
    images = []
    labels = []
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if os.path.isdir(label_path):
            count = 0
            for image_file in os.listdir(label_path):
                if limit_per_class and count >= limit_per_class:
                    break
                image_path = os.path.join(label_path, image_file)
                print(f'Loading {image_path}')
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                images.append(image)
                labels.append(label)
                count += 1
    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Encode labels
    class_labels = sorted(set(labels))
    label_map = {label: idx for idx, label in enumerate(class_labels)}
    labels = np.array([label_map[label] for label in labels])

    return images, labels, class_labels

# Load the dataset with a limit of 500 images per class
print('Loading dataset...')
images, labels, class_labels = load_data(data_dir, limit_per_class=1000)
print(f'Dataset loaded with {len(images)} images.')

# Check if any images were loaded
if len(images) == 0:
    print('No images were loaded. Please check your dataset directory.')
else:
    # Normalize images
    images = images / 255.0

    # One-hot encode labels
    labels = to_categorical(labels, num_classes=len(class_labels))

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Add channel dimension
    X_train = np.expand_dims(X_train, axis=-1)
    X_val = np.expand_dims(X_val, axis=-1)

    print(f'Training data shape: {X_train.shape}')
    print(f'Validation data shape: {X_val.shape}')
    print(f'Number of classes: {len(class_labels)}')

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(X_train)

    # Define the model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(class_labels), activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Display model architecture
    model.summary()

    # Define callbacks
    checkpoint = ModelCheckpoint('isl_model.keras', monitor='val_loss', save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')

    # Train the model
    history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                        epochs=25,  # Reduced epochs for quick testing
                        validation_data=(X_val, y_val),
                        callbacks=[checkpoint, early_stopping])

    # Load the best model
    model.load_weights('isl_model.keras')

    # Evaluate on validation set
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f'Validation Accuracy: {val_acc:.4f}')
