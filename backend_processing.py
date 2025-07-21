import os
import cv2
import numpy as np
import tensorflow as tf
import logging
from mtcnn import MTCNN
try:
    from preprocessing_videos import extract_features as extract_video_features
    from preprocessing_images import extract_image_features
    logging.info("Successfully imported feature extractors from preprocessing scripts.")
except ImportError as e:
    logging.critical(f"Failed to import preprocessing functions. Ensure preprocessing_videos.py and preprocessing_images.py are available. Error: {e}")
    def extract_video_features(*args, **kwargs): return None, None, None
    def extract_image_features(*args, **kwargs): return None, None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MTCNN detector
mtcnn_detector = MTCNN()

# Load both pre-trained models
try:
    logger.info("Loading deepfake detection models...")
    video_model_path = os.path.join('models', 'Xception_model.h5')
    image_model_path = os.path.join('models', 'Xception_image_model.h5')

    if not os.path.exists(video_model_path):
        raise FileNotFoundError(f"Video model not found at: {video_model_path}")
    if not os.path.exists(image_model_path):
        raise FileNotFoundError(f"Image model not found at: {image_model_path}")

    video_model = tf.keras.models.load_model(video_model_path, compile=False)
    image_model = tf.keras.models.load_model(image_model_path, compile=False)
    logger.info("Models loaded successfully.")
except Exception as e:
    logger.critical(f"CRITICAL ERROR: Failed to load models. The application will not work. {e}")
    video_model = None
    image_model = None


STANDARD_SIZE = (299, 299) # Match Xception input size
FEATURE_METRICS_THRESHOLD = 8750000.0


def _uniform_frame_sampling(video_path, num_frames):
    """Selects equally spaced frames from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video for sampling: {video_path}")
        return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total_frames <= 0:
        return []
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    return indices.tolist()

def generate_transformed_video(video_path, pred_frame_idxs, predictions, face_boxes):
    """Generates a video with deepfake probability and a bounding box overlaid on frames."""
    output_dir = "transformed_videos"
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(video_path)
    output_path = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}_transformed.mp4")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open original video for transformation: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if width == 0 or height == 0:
        logger.error(f"Invalid video dimensions ({width}x{height}) for: {video_path}")
        cap.release()
        return None

    fourcc_h264 = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc_h264, fps, (width, height))

    if not out.isOpened():
        logger.warning("Failed to open VideoWriter with 'avc1' (H.264) codec. The required OpenH264 library may be missing.")
        logger.warning("Falling back to 'mp4v' codec. This may have browser compatibility issues.")
        fourcc_mp4v = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc_mp4v, fps, (width, height))
        
        if not out.isOpened():
            logger.error("Failed to open VideoWriter with fallback codec 'mp4v'. Video creation failed.")
            cap.release()
            return None

    pred_data_dict = dict(zip(pred_frame_idxs, zip(predictions, face_boxes)))
    current_prob = -1
    current_box = None

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in pred_data_dict:
            current_prob, current_box = pred_data_dict[frame_idx]

        if current_prob != -1:
            text = f"Deepfake Probability: {current_prob:.2f}"
            color = (0, 0, 255) if current_prob >= 0.5 else (0, 255, 0)
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            if current_box:
                x, y, w, h = current_box
                if w > 0 and h > 0:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        out.write(frame)

    cap.release()
    out.release()
    logger.info(f"Generated transformed video at {output_path}")
    return output_path

def generate_gradcam_heatmap(model, image_array, feature_array, original_frame, face_box):
    """Generates and overlays a Grad-CAM heatmap on the original frame."""
    try:
        image_input = np.expand_dims(image_array, axis=0)
        feature_input = np.expand_dims(feature_array, axis=0)

        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer('block14_sepconv2_act').output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model([image_input, feature_input])
            loss = predictions[0]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
        heatmap = cv2.resize(heatmap, (face_box[2], face_box[3]))
        heatmap = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        x, y, w, h = face_box
        face_region = original_frame[y:y+h, x:x+w]
        blended_face = cv2.addWeighted(face_region, 0.6, heatmap_colored, 0.4, 0)
        
        temp_dir = "gradcam_output"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"gradcam_{np.random.randint(100000)}.jpg")
        
        cv2.imwrite(temp_path, blended_face)
        return temp_path
    except Exception as e:
        logger.error(f"Error in Grad-CAM generation: {e}")
        return None

def analyze_video(video_path, num_keyframes=3, num_pred_frames=10):
    """Analyzes a video for deepfake detection using the video model."""
    if video_model is None:
        return "Error: Video model not loaded.", 0.0, [], [], None, [], None
    
    per_frame_features, avg_gan_fingerprint, key_frame_idxs = extract_video_features(
        video_path, num_frames=num_keyframes, is_image=False
    )
    
    if key_frame_idxs is None or not key_frame_idxs:
        return "Error: Could not extract video features.", 0.0, [], [], None, [], None

    pred_frame_idxs = _uniform_frame_sampling(video_path, num_pred_frames)
    if not pred_frame_idxs:
        return "Error: Could not sample frames for prediction.", 0.0, [], [], None, [], None

    assigned_features = [per_frame_features[np.argmin(np.abs(np.array(key_frame_idxs) - p))] for p in pred_frame_idxs]
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error: Could not open video to extract frames.", 0.0, [], [], None, [], None
        
    images, original_frames, face_boxes = [], [], []
    for pred_idx in pred_frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pred_idx)
        ret, frame = cap.read()
        if not ret: continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = mtcnn_detector.detect_faces(frame_rgb)
        box = faces[0]['box'] if faces else [0, 0, frame.shape[1], frame.shape[0]]
        
        face = frame_rgb[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
        images.append(cv2.resize(face, STANDARD_SIZE) / 255.0)
        original_frames.append(frame)
        face_boxes.append(box)
    cap.release()

    if not images:
        return "Error: No faces detected in sampled frames.", 0.0, [], [], None, [], None

    predictions = video_model.predict([np.array(images), np.array(assigned_features)], verbose=0).flatten()
    avg_prob = np.mean(predictions)
    prediction = "Deepfake" if avg_prob >= 0.5 else "Real"
    confidence = (avg_prob if prediction == "Deepfake" else 1 - avg_prob) * 100
    
    transformed_video_path = generate_transformed_video(video_path, pred_frame_idxs, predictions, face_boxes)
    gradcam_paths = []
    if prediction == "Deepfake":
        for img_arr, feat_arr, orig_frame, box in zip(images, assigned_features, original_frames, face_boxes):
            path = generate_gradcam_heatmap(video_model, img_arr, feat_arr, orig_frame, box)
            if path: gradcam_paths.append(path)

    feature_metrics_for_ui = avg_gan_fingerprint.tolist() if avg_gan_fingerprint is not None else []
    return prediction, float(confidence), predictions.tolist(), feature_metrics_for_ui, transformed_video_path, gradcam_paths, FEATURE_METRICS_THRESHOLD

def analyze_image(image_path):
    """Analyzes an image for deepfake detection using the image model."""
    if image_model is None:
        return "Error: Image model not loaded.", 0.0, [], [], None, None
    
    features, gan_fingerprint = extract_image_features(image_path)
    
    if features is None:
        return "Error: Could not extract image features.", 0.0, [], [], None, None

    original_image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    faces = mtcnn_detector.detect_faces(image_rgb)
    box = faces[0]['box'] if faces else [0, 0, original_image.shape[1], original_image.shape[0]]
    face = image_rgb[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
    
    image_data = cv2.resize(face, STANDARD_SIZE) / 255.0
    
    prediction_prob = image_model.predict([np.expand_dims(image_data, axis=0), np.expand_dims(features, axis=0)], verbose=0)[0][0]
    prediction = "Deepfake" if prediction_prob >= 0.5 else "Real"
    confidence = (prediction_prob if prediction == "Deepfake" else 1 - prediction_prob) * 100
    
    gradcam_path = None
    if prediction == "Deepfake":
        gradcam_path = generate_gradcam_heatmap(image_model, image_data, features, original_image, box)
    
    feature_metrics_for_ui = features.tolist() if features is not None else []
    return prediction, float(confidence), [float(prediction_prob)], feature_metrics_for_ui, gradcam_path, FEATURE_METRICS_THRESHOLD