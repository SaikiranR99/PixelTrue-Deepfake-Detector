import tensorflow as tf
from tensorflow.keras.applications import Xception # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Concatenate, Dropout# type: ignore
from tensorflow.keras.models import Model# type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator# type: ignore
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping# type: ignore
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
data_dir = r'images'
output_dir = r'models'
# Directory for features extracted by preprocessing.py
feature_dir = r'split_dataset_output/per_frame_features' 
feature_size = 100 # Must match the output size in preprocessing.py

os.makedirs(output_dir, exist_ok=True)

# Load image list and split
csv_file = r"data/images.csv"
if not os.path.exists(csv_file):
    logger.error(f"CSV file not found at: {csv_file}")
else:
    df = pd.read_csv(csv_file)
    df['label'] = df['class'].astype(str) # Ensure labels are strings for the generator
    
    # Add full path to filenames
    df['filename'] = df['filename'].apply(lambda x: os.path.join(data_dir, x))

    # Split data: 80% train, 10% validation, 10% test
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

    logger.info(f"Training samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")
    logger.info(f"Test samples: {len(test_df)}")

# Function to get features for a given image filename
def get_features(image_filename, feature_dir):
    base_name = os.path.basename(image_filename)
    feature_name = os.path.splitext(base_name)[0] + '.npy'
    feature_path = os.path.join(feature_dir, feature_name)
    
    if not os.path.exists(feature_path):
        logger.warning(f"Feature file not found for image {base_name}, returning zeros.")
        return np.zeros(feature_size)
    return np.load(feature_path)

# Data augmentation and generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2]
)
val_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 32

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='filename',
    y_col='label',
    target_size=(299, 299),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

val_generator = val_datagen.flow_from_dataframe(
    val_df,
    x_col='filename',
    y_col='label',
    target_size=(299, 299),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

test_generator = val_datagen.flow_from_dataframe(
    test_df,
    x_col='filename',
    y_col='label',
    target_size=(299, 299),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Custom data generator to yield images and their features
def custom_generator(image_generator, feature_dir):
    while True:
        for i in range(len(image_generator)):
            x_batch_img, y_batch = image_generator[i]
            
            # Get filenames for the current batch
            start_index = i * image_generator.batch_size
            end_index = start_index + len(x_batch_img)
            batch_filenames = image_generator.filenames[start_index:end_index]

            # Load features for the batch
            x_batch_features = np.array([get_features(f, feature_dir) for f in batch_filenames])
            
            yield (x_batch_img, x_batch_features), y_batch

train_dataset = tf.data.Dataset.from_generator(
    lambda: custom_generator(train_generator, feature_dir),
    output_signature=(
        (tf.TensorSpec(shape=(None, 299, 299, 3), dtype=tf.float32),
         tf.TensorSpec(shape=(None, feature_size), dtype=tf.float32)),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )
).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
    lambda: custom_generator(val_generator, feature_dir),
    output_signature=(
        (tf.TensorSpec(shape=(None, 299, 299, 3), dtype=tf.float32),
         tf.TensorSpec(shape=(None, feature_size), dtype=tf.float32)),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )
)

# Build model
def build_model(input_shape=(299, 299, 3), feature_size=100):
    # Image input branch
    image_input = Input(shape=input_shape, name='image_input')
    base_model = Xception(weights='imagenet', include_top=False, input_tensor=image_input)
    base_model.trainable = True # Fine-tune the model
    
    # Freeze only the first few layers
    for layer in base_model.layers[:40]:
        layer.trainable = False
        
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    
    # Feature input branch
    feature_input = Input(shape=(feature_size,), name='feature_input')
    f = Dense(256, activation='relu')(feature_input)
    f = Dropout(0.5)(f)
    
    # Combine branches
    combined = Concatenate()([x, f])
    combined = Dense(1024, activation='relu')(combined)
    combined = Dropout(0.5)(combined)
    predictions = Dense(1, activation='sigmoid')(combined)
    
    model = Model(inputs=[image_input, feature_input], outputs=predictions)
    return model

# Compile and train
model = build_model(feature_size=feature_size)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6, verbose=1)
    ]
)

# Save the final model
model.save(os.path.join(output_dir, 'Xception_image_model.h5'))
logger.info(f"Model saved to {os.path.join(output_dir, 'Xception_image_model.h5')}")

# Evaluate on the test set
test_dataset = tf.data.Dataset.from_generator(
    lambda: custom_generator(test_generator, feature_dir),
    output_signature=(
        (tf.TensorSpec(shape=(None, 299, 299, 3), dtype=tf.float32),
         tf.TensorSpec(shape=(None, feature_size), dtype=tf.float32)),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )
)

logger.info("Evaluating on the test set...")
test_predictions = model.predict(test_dataset, steps=len(test_generator)).flatten()
y_pred_test = (test_predictions > 0.5).astype(int)
test_labels = test_generator.classes
accuracy_test = accuracy_score(test_labels, y_pred_test)
logger.info(f"Test Accuracy: {accuracy_test:.4f}")

# Plotting functions
def plot_training_history(history, output_dir):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

plot_training_history(history, output_dir)
plot_confusion_matrix(test_labels, y_pred_test, output_dir)
logger.info("Training history and confusion matrix plots saved.")