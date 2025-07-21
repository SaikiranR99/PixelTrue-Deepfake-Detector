import os
import cv2
import numpy as np
import pandas as pd
import logging
from mtcnn import MTCNN
from scipy.fft import dct
from pywt import dwt2, WaveletPacket2D
import torch
from torchvision.models import mobilenet_v2
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('preprocessing.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Define paths
image_dir = r"images"
csv_path = r"data/images.csv"
features_dirs = {
    'gan_fingerprint': r"split_dataset_output/gan_fingerprint_features",
    'per_frame': r"split_dataset_output/per_frame_features" # Retaining name for consistency
}

# Ensure directories exist
for key in features_dirs:
    os.makedirs(features_dirs[key], exist_ok=True)
    logger.info(f"Ensured directory exists: {features_dirs[key]}")

# Load detectors and models
logger.info("Loading MTCNN and MobileNetV2...")
mtcnn_detector = MTCNN()
mobilenet = mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT').eval()
if torch.cuda.is_available():
    mobilenet = mobilenet.cuda()
    logger.info("MobileNetV2 moved to GPU.")

def extract_image_features(image_path, size=(64, 64)):
    """Extract features from a single image."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Failed to read image {image_path}")
            return None, None

        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = mtcnn_detector.detect_faces(frame_rgb)
        
        if faces:
            x1, y1, width, height = faces[0]['box']
            x2, y2 = x1 + width, y1 + height
            face = frame_rgb[max(0, y1):y2, max(0, x1):x2]
        else:
            logger.warning(f"No face detected in {image_path}, using full image.")
            face = frame_rgb

        # --- CNN Features ---
        face_cnn = cv2.resize(face, (224, 224))
        face_tensor = torch.tensor(face_cnn.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
        if torch.cuda.is_available():
            face_tensor = face_tensor.cuda()
        with torch.no_grad():
            cnn_features = mobilenet(face_tensor).cpu().numpy().flatten()

        # --- Grayscale Features ---
        face_small = cv2.resize(face, size)
        face_gray = cv2.cvtColor(face_small, cv2.COLOR_RGB2GRAY)

        # --- Feature Extraction ---
        laplacian = cv2.Laplacian(face_gray, cv2.CV_64F).flatten() / 255.0
        
        dct_coeffs = dct(dct(face_gray.T, norm='ortho').T, norm='ortho').flatten()
        dct_features = dct_coeffs[:100]

        f = np.fft.fft2(face_gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.log1p(np.abs(fshift))
        rows, cols = magnitude_spectrum.shape
        crow, ccol = rows // 2, cols // 2
        
        x_coords, y_coords = np.arange(cols) - ccol, np.arange(rows) - crow
        xx, yy = np.meshgrid(x_coords, y_coords)
        dist = np.sqrt(xx**2 + yy**2)
        max_dist = np.max(dist)
        
        num_bins = 20
        bins = np.linspace(0, max_dist, num_bins + 1)
        azimuthal_avg = [
            np.mean(magnitude_spectrum[(dist >= bins[i-1]) & (dist < bins[i])])
            if np.any((dist >= bins[i-1]) & (dist < bins[i])) else 0.0
            for i in range(1, num_bins + 1)
        ]
        
        gan_fingerprint = np.concatenate([laplacian, cnn_features, dct_features, np.array(azimuthal_avg)])

        # --- UCF Features ---
        coeffs, _ = dwt2(face_gray, 'db1')
        wp = WaveletPacket2D(data=face_gray, wavelet='db1', mode='symmetric', maxlevel=2)
        wavelet_features = np.concatenate([coeffs.flatten(), wp['aa'].data.flatten()])
        ucf_features = np.concatenate([wavelet_features, [np.mean(dct_coeffs), np.std(dct_coeffs)]])

        # --- CORRECTED: Combined and Reduced Features ---
        combined_features = np.concatenate([gan_fingerprint, ucf_features])
        
        target_size = 100
        if len(combined_features) >= target_size:
            # Truncate if longer than target size
            features_transformed = combined_features[:target_size]
        else:
            # Pad with zeros if shorter
            padded_features = np.zeros(target_size, dtype=np.float32)
            padded_features[:len(combined_features)] = combined_features
            features_transformed = padded_features
            
        return features_transformed, gan_fingerprint

    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        return None, None

def save_features_for_images(image_list, base_dir, directories):
    """Iterate through images, extract features, and save them."""
    logger.info("Starting feature extraction for all images...")
    for filename in tqdm(image_list, desc="Processing images"):
        full_path = os.path.join(base_dir, filename)
        if not os.path.exists(full_path):
            logger.warning(f"Image file {full_path} not found.")
            continue
        
        per_image_feats, gan_fingerprint = extract_image_features(full_path)
        
        if per_image_feats is not None and gan_fingerprint is not None:
            image_name = os.path.splitext(filename)[0]
            
            # Save per-image features (used in the training generator)
            feature_path = os.path.join(directories['per_frame'], f"{image_name}.npy")
            np.save(feature_path, per_image_feats)
            
            # Save GAN fingerprint features
            gan_path = os.path.join(directories['gan_fingerprint'], f"{image_name}.npy")
            np.save(gan_path, gan_fingerprint)
            
    logger.info("Feature extraction completed.")

if __name__ == "__main__":
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found at {csv_path}. Please check the path.")
    else:
        image_df = pd.read_csv(csv_path)
        image_filenames = image_df["filename"].tolist()
        save_features_for_images(image_filenames, image_dir, features_dirs)