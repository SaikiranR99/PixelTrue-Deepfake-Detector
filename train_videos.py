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
data_dir = r'data\faces'
output_dir = r'models'
fingerprint_dir = r'split_dataset_output\gan_fingerprint_features'
per_frame_dir = r'split_dataset_output\per_frame_features'
feature_size = 100

os.makedirs(output_dir, exist_ok=True)

# Load video list and split
csv_file = r"data\videos.csv"
video_list = pd.read_csv(csv_file)

# First split: 80% train, 20% temp
train_videos, temp_videos = train_test_split(video_list['filename'], test_size=0.2, random_state=42)
# Second split: 50% of temp for val and 50% for test (10% each of total)
val_videos, test_videos = train_test_split(temp_videos, test_size=0.5, random_state=42)

logger.info(f"Number of train videos: {len(train_videos)}")
logger.info(f"Number of val videos: {len(val_videos)}")
logger.info(f"Number of test videos: {len(test_videos)}")

# Function to get features from the closest key frame
def get_features(video_name, frame_id, per_frame_dir):
    frame_id = int(frame_id)
    keyframe_file = os.path.join(per_frame_dir, f"{video_name}_keyframes.npy")
    if not os.path.exists(keyframe_file):
        logger.warning(f"Keyframe file not found for {video_name}")
        return np.zeros(feature_size)
    keyframe_indices = np.load(keyframe_file)
    if len(keyframe_indices) == 0:
        logger.warning(f"No keyframe indices for {video_name}")
        return np.zeros(feature_size)
    closest_keyframe_idx = np.argmin(np.abs(keyframe_indices - frame_id))
    feature_index = closest_keyframe_idx
    per_frame_path = os.path.join(per_frame_dir, f"{video_name}_frame{feature_index}.npy")
    if not os.path.exists(per_frame_path):
        logger.warning(f"Feature file not found: {per_frame_path}")
        return np.zeros(feature_size)
    return np.load(per_frame_path)

# Function to get video filename from frame filename
def get_video_filename(frame_filename):
    base = os.path.basename(frame_filename)
    parts = base.split('_')
    video_name = '_'.join(parts[:-1])
    return video_name + '.mp4'

# Prepare image paths and labels
image_paths = []
labels = []
for _, row in video_list.iterrows():
    video_name = os.path.splitext(row['filename'])[0]
    label = str(row['label'])
    frame_images = [f for f in os.listdir(data_dir) if f.startswith(video_name + '_') and f.endswith('.jpg')]
    for frame in frame_images:
        image_paths.append(os.path.join(data_dir, frame))
        labels.append(label)

df = pd.DataFrame({'filename': image_paths, 'class': labels})

# Split data
train_videos_set = set(train_videos)
val_videos_set = set(val_videos)
test_videos_set = set(test_videos)
train_df = df[df['filename'].apply(lambda x: get_video_filename(x) in train_videos_set)]
val_df = df[df['filename'].apply(lambda x: get_video_filename(x) in val_videos_set)]
test_df = df[df['filename'].apply(lambda x: get_video_filename(x) in test_videos_set)]

logger.info(f"Number of training samples: {len(train_df)}")
logger.info(f"Number of validation samples: {len(val_df)}")
logger.info(f"Number of test samples: {len(test_df)}")
logger.info(f"Unique classes in train_df: {train_df['class'].unique()}")
logger.info(f"Unique classes in val_df: {val_df['class'].unique()}")

assert len(train_df['class'].unique()) == 2, "Training data must have exactly two classes."
assert len(val_df['class'].unique()) == 2, "Validation data must have exactly two classes."

# Compute class weights
num_class0 = (train_df['class'] == '0').sum()
num_class1 = (train_df['class'] == '1').sum()
total = len(train_df)
weight0 = total / (2.0 * num_class0) if num_class0 > 0 else 1.0
weight1 = total / (2.0 * num_class1) if num_class1 > 0 else 1.0
logger.info(f"Class distribution: class 0: {num_class0}, class 1: {num_class1}")
logger.info(f"Class weights: class 0: {weight0:.4f}, class 1: {weight1:.4f}")

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2]
)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='filename',
    y_col='class',
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary',
    shuffle=True
)

val_generator = val_datagen.flow_from_dataframe(
    val_df,
    x_col='filename',
    y_col='class',
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col='filename',
    y_col='class',
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Load features
train_features = []
for f in train_generator.filenames:
    video_name = '_'.join(os.path.basename(f).split('_')[:-1])
    frame_id = os.path.basename(f).split('_')[-1].split('.')[0]
    feature = get_features(video_name, frame_id, per_frame_dir)
    train_features.append(feature)
train_features = np.array(train_features)

val_features = []
for f in val_generator.filenames:
    video_name = '_'.join(os.path.basename(f).split('_')[:-1])
    frame_id = os.path.basename(f).split('_')[-1].split('.')[0]
    feature = get_features(video_name, frame_id, per_frame_dir)
    val_features.append(feature)
val_features = np.array(val_features)

test_features = []
for f in test_generator.filenames:
    video_name = '_'.join(os.path.basename(f).split('_')[:-1])
    frame_id = os.path.basename(f).split('_')[-1].split('.')[0]
    feature = get_features(video_name, frame_id, per_frame_dir)
    test_features.append(feature)
test_features = np.array(test_features)

def custom_generator(image_generator, features):
    def gen():
        while True:  # Loop indefinitely to prevent running out of data
            for i in range(len(image_generator)):
                x_batch, y_batch = image_generator[i]
                feature_batch = features[i * image_generator.batch_size:(i + 1) * image_generator.batch_size]
                yield (x_batch, feature_batch), y_batch
    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            (
                tf.TensorSpec(shape=(None, 299, 299, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, feature_size), dtype=tf.float32)
            ),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)

train_dataset = custom_generator(train_generator, train_features)
val_dataset = custom_generator(val_generator, val_features)
test_dataset = custom_generator(test_generator, test_features)

# Build model
def build_model(input_shape=(299, 299, 3), feature_size=100):
    image_input = Input(shape=input_shape)
    base_model = Xception(weights='imagenet', include_top=False, input_tensor=image_input)
    
    num_layers = len(base_model.layers)
    num_to_unfreeze = max(1, int(0.1 * num_layers))  # Ensure at least 1 layer is unfrozen
    logger.info(f"Total layers: {num_layers}, Unfreezing last {num_to_unfreeze} layers")
    
    for layer in base_model.layers[:-num_to_unfreeze]:
        layer.trainable = False
    for layer in base_model.layers[-num_to_unfreeze:]:
        layer.trainable = True
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    
    feature_input = Input(shape=(feature_size,))
    f = Dense(256, activation='relu')(feature_input)
    f = Dropout(0.5)(f)
    
    combined = Concatenate()([x, f])
    combined = Dense(1024, activation='relu')(combined)
    combined = Dropout(0.5)(combined)
    predictions = Dense(1, activation='sigmoid')(combined)
    
    model = Model(inputs=[image_input, feature_input], outputs=predictions)
    return model

# Custom loss function
def weighted_binary_crossentropy(weight0, weight1):
    def loss(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        weight_vector = y_true * weight1 + (1 - y_true) * weight0
        weighted_bce = weight_vector * bce
        return tf.reduce_mean(weighted_bce)
    return loss

# Compile and train
model = build_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=weighted_binary_crossentropy(weight0, weight1),
              metrics=['accuracy'])

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,  # Adjusted to match your query
    steps_per_epoch=len(train_generator),  # 933 steps as per your log
    validation_steps=len(val_generator),
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)
    ],
    verbose=1
)
model.save(os.path.join(output_dir, 'Xception_model.h5'))

# Quantize and save
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open(os.path.join(output_dir, 'Xception_quantized.tflite'), 'wb') as f:
    f.write(tflite_model)

# Evaluate on test set
test_pred_gen = custom_generator(test_generator, test_features)
test_predictions = model.predict(test_pred_gen, steps=len(test_generator)).flatten()
test_labels = test_generator.classes
y_pred_test = (test_predictions > 0.5).astype(int)
accuracy_test = accuracy_score(test_labels, y_pred_test)
logger.info(f"Test Accuracy: {accuracy_test:.4f}")

# Plot results
def plot_training_history(history, output_dir):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

plot_training_history(history, output_dir)
plot_confusion_matrix(test_labels, y_pred_test, output_dir)