import os
import cv2
import numpy as np
import pandas as pd
import logging
from mtcnn import MTCNN
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from scipy.fft import dct
from pywt import dwt2, WaveletPacket2D
from sklearn.decomposition import PCA
import torch
from torchvision.models import mobilenet_v2
import multiprocessing
from scenedetect import open_video, SceneManager, ContentDetector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('preprocessing.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Define paths
video_dir = r"videos"
faces_dir = r"data\faces"
features_dirs = {
    'gan_fingerprint': r"split_dataset_output\gan_fingerprint_features",
    'per_frame': r"split_dataset_output\per_frame_features"
}

# Ensure directories exist
if not os.path.exists(faces_dir):
    os.makedirs(faces_dir)
    logger.info(f"Created directory: {faces_dir}")
for key in features_dirs:
    os.makedirs(features_dirs[key], exist_ok=True)

# Load detectors and models
logger.info("Loading MTCNN and MobileNetV2...")
mtcnn_detector = MTCNN()
mobilenet = mobilenet_v2(pretrained=True).eval()
if torch.cuda.is_available():
    mobilenet = mobilenet.cuda()

def get_adaptive_frames(video_path, num_frames=3):
    """Select adaptive frames from a video."""
    try:
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector())
        scene_manager.detect_scenes(video=video)
        scene_list = scene_manager.get_scene_list()
        
        logger.info(f"Detected {len(scene_list)} scenes in {video_path}")
        
        if len(scene_list) == 0:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if total_frames > 0:
                step = total_frames // num_frames
                selected_frames = [i * step for i in range(num_frames)]
                selected_frames[-1] = min(selected_frames[-1], total_frames - 1)
                return selected_frames
            return []
        
        frame_numbers = [scene[0].get_frames() for scene in scene_list]
        if len(frame_numbers) >= num_frames:
            return frame_numbers[:num_frames]
        selected_frames = frame_numbers
        while len(selected_frames) < num_frames:
            selected_frames.append(frame_numbers[-1])
        return selected_frames
    except Exception as e:
        logger.error(f"Error in get_adaptive_frames for {video_path}: {e}")
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total_frames > 0:
            step = total_frames // num_frames
            selected_frames = [i * step for i in range(num_frames)]
            selected_frames[-1] = min(selected_frames[-1], total_frames - 1)
            return selected_frames
        return []

def extract_features(input_path, size=(64, 64), num_frames=3, is_image=False):
    """Extract features from video or image input."""
    if is_image:
        image = cv2.imread(input_path)
        if image is None:
            logger.warning(f"Failed to read image {input_path}")
            return None, None, []
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = mtcnn_detector.detect_faces(frame_rgb)
        if faces:
            face = frame_rgb[max(0, faces[0]['box'][1]):max(0, faces[0]['box'][1] + faces[0]['box'][3]),
                             max(0, faces[0]['box'][0]):max(0, faces[0]['box'][0] + faces[0]['box'][2])]
        else:
            face = frame_rgb
        
        face_cnn = cv2.resize(face, (224, 224))
        face_tensor = torch.tensor(face_cnn.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
        if torch.cuda.is_available():
            face_tensor = face_tensor.cuda()
        with torch.no_grad():
            cnn_features = mobilenet(face_tensor).cpu().numpy().flatten()
        
        face_small = cv2.resize(face, size)
        face_gray = cv2.cvtColor(face_small, cv2.COLOR_RGB2GRAY)
        
        laplacian = cv2.Laplacian(face_gray, cv2.CV_64F).flatten() / 255.0
        dct_coeffs = dct(dct(face_gray.T, norm='ortho').T, norm='ortho').flatten()
        dct_features = dct_coeffs[:100]
        
        f = np.fft.fft2(face_gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.log1p(np.abs(fshift))
        rows, cols = magnitude_spectrum.shape
        crow, ccol = rows // 2, cols // 2
        x, y = np.arange(cols) - ccol, np.arange(rows) - crow
        xx, yy = np.meshgrid(x, y)
        dist = np.sqrt(xx**2 + yy**2)
        max_dist = np.max(dist)
        num_bins = 20
        bins = np.linspace(0, max_dist, num_bins + 1)
        azimuthal_avg = [np.mean(magnitude_spectrum[(dist >= bins[i-1]) & (dist < bins[i])]) if np.any((dist >= bins[i-1]) & (dist < bins[i])) else 0.0 for i in range(1, num_bins + 1)]
        
        gan_fingerprint = np.concatenate([laplacian, cnn_features, dct_features, np.array(azimuthal_avg)])
        
        coeffs, _ = dwt2(face_gray, 'db1')
        wp = WaveletPacket2D(data=face_gray, wavelet='db1', mode='symmetric', maxlevel=2)
        wavelet_features = np.concatenate([coeffs.flatten(), wp['aa'].data.flatten()])
        ucf_features = np.concatenate([wavelet_features, [np.mean(dct_coeffs), np.std(dct_coeffs)]])
        
        per_frame_features = np.concatenate([gan_fingerprint, ucf_features])
        n_samples = 1
        n_components = min(100, n_samples)
        pca = PCA(n_components=n_components)
        features_transformed = pca.fit_transform(per_frame_features.reshape(1, -1))[0]
        target_size = 100
        if len(features_transformed) < target_size:
            padded_features = np.zeros(target_size, dtype=np.float32)
            padded_features[:len(features_transformed)] = features_transformed
            features_transformed = padded_features
        
        return features_transformed, np.zeros(100, dtype=np.float32), [0]
    
    else:
        frame_idxs = get_adaptive_frames(input_path, num_frames)
        if not frame_idxs:
            logger.warning(f"No frames selected for {input_path}")
            return np.zeros((num_frames, 100), dtype=np.float32), np.zeros(100, dtype=np.float32), []
        
        cap = cv2.VideoCapture(input_path)
        video_name = os.path.splitext(os.path.basename(input_path))[0]
        per_frame_features, gan_fingerprints = [], []
        
        for i, idx in enumerate(frame_idxs):
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.warning(f"Failed to read frame {idx} from {input_path}")
                    continue
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = mtcnn_detector.detect_faces(frame_rgb)
                if faces:
                    face = frame_rgb[max(0, faces[0]['box'][1]):max(0, faces[0]['box'][1] + faces[0]['box'][3]),
                                     max(0, faces[0]['box'][0]):max(0, faces[0]['box'][0] + faces[0]['box'][2])]
                else:
                    face = frame_rgb
                
                face_cnn = cv2.resize(face, (224, 224))
                face_tensor = torch.tensor(face_cnn.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
                if torch.cuda.is_available():
                    face_tensor = face_tensor.cuda()
                with torch.no_grad():
                    cnn_features = mobilenet(face_tensor).cpu().numpy().flatten()
                
                face_small = cv2.resize(face, size)
                face_gray = cv2.cvtColor(face_small, cv2.COLOR_RGB2GRAY)
                
                laplacian = cv2.Laplacian(face_gray, cv2.CV_64F).flatten() / 255.0
                dct_coeffs = dct(dct(face_gray.T, norm='ortho').T, norm='ortho').flatten()
                dct_features = dct_coeffs[:100]
                
                f = np.fft.fft2(face_gray)
                fshift = np.fft.fftshift(f)
                magnitude_spectrum = np.log1p(np.abs(fshift))
                rows, cols = magnitude_spectrum.shape
                crow, ccol = rows // 2, cols // 2
                x, y = np.arange(cols) - ccol, np.arange(rows) - crow
                xx, yy = np.meshgrid(x, y)
                dist = np.sqrt(xx**2 + yy**2)
                max_dist = np.max(dist)
                num_bins = 20
                bins = np.linspace(0, max_dist, num_bins + 1)
                azimuthal_avg = [np.mean(magnitude_spectrum[(dist >= bins[i-1]) & (dist < bins[i])]) if np.any((dist >= bins[i-1]) & (dist < bins[i])) else 0.0 for i in range(1, num_bins + 1)]
                
                gan_fingerprint = np.concatenate([laplacian, cnn_features, dct_features, np.array(azimuthal_avg)])
                gan_fingerprints.append(gan_fingerprint)
                
                coeffs, _ = dwt2(face_gray, 'db1')
                wp = WaveletPacket2D(data=face_gray, wavelet='db1', mode='symmetric', maxlevel=2)
                wavelet_features = np.concatenate([coeffs.flatten(), wp['aa'].data.flatten()])
                ucf_features = np.concatenate([wavelet_features, [np.mean(dct_coeffs), np.std(dct_coeffs)]])
                
                per_frame_features.append(np.concatenate([gan_fingerprint, ucf_features]))
            except Exception as e:
                logger.error(f"Error processing frame {idx} in {input_path}: {e}")
        
        cap.release()
        
        if per_frame_features:
            try:
                n_samples = len(per_frame_features)
                n_features = per_frame_features[0].shape[0]
                n_components_per_frame = min(100, n_samples, n_features)
                pca_per_frame = PCA(n_components=n_components_per_frame)
                per_frame_features_transformed = pca_per_frame.fit_transform(np.array(per_frame_features))
                
                gan_fingerprints_array = np.array(gan_fingerprints)
                n_features_gan = gan_fingerprints_array.shape[1]
                n_components_gan = min(100, n_samples, n_features_gan)
                pca_gan = PCA(n_components=n_components_gan)
                gan_fingerprints_transformed = pca_gan.fit_transform(gan_fingerprints_array)
                avg_gan_fingerprint = np.mean(gan_fingerprints_transformed, axis=0)
                
                target_size = 100
                if n_components_per_frame < target_size:
                    padded_features = np.zeros((per_frame_features_transformed.shape[0], target_size), dtype=np.float32)
                    padded_features[:, :n_components_per_frame] = per_frame_features_transformed
                    per_frame_features_transformed = padded_features
                elif n_components_per_frame > target_size:
                    per_frame_features_transformed = per_frame_features_transformed[:, :target_size]
                
                if n_components_gan < target_size:
                    padded_gan = np.zeros(target_size, dtype=np.float32)
                    padded_gan[:n_components_gan] = avg_gan_fingerprint
                    avg_gan_fingerprint = padded_gan
                elif n_components_gan > target_size:
                    avg_gan_fingerprint = avg_gan_fingerprint[:target_size]
            except Exception as e:
                logger.error(f"Error in dimensionality reduction for {input_path}: {e}")
                avg_gan_fingerprint = np.zeros(100, dtype=np.float32)
                per_frame_features_transformed = np.zeros((num_frames, 100), dtype=np.float32)
        else:
            avg_gan_fingerprint = np.zeros(100, dtype=np.float32)
            per_frame_features_transformed = np.zeros((num_frames, 100), dtype=np.float32)
        
        return per_frame_features_transformed, avg_gan_fingerprint, frame_idxs

def process_video(video_path, directories, video_dir, faces_dir):
    full_path = os.path.join(video_dir, video_path)
    if not os.path.exists(full_path):
        logger.warning(f"Video file {video_path} not found.")
        return
    try:
        per_frame_feats, avg_gan_fingerprint, frame_idxs = extract_features(full_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        for i, feat in enumerate(per_frame_feats):
            feature_path = os.path.join(directories['per_frame'], f"{video_name}_frame{i}.npy")
            np.save(feature_path, feat)
            logger.info(f"Saved feature: {feature_path}")
        gan_path = os.path.join(directories['gan_fingerprint'], f"{video_name}.npy")
        np.save(gan_path, avg_gan_fingerprint)
        logger.info(f"Saved GAN fingerprint: {gan_path}")
        keyframe_path = os.path.join(directories['per_frame'], f"{video_name}_keyframes.npy")
        np.save(keyframe_path, frame_idxs)
        logger.info(f"Saved keyframe indices: {keyframe_path}")
    except Exception as e:
        logger.error(f"Error processing video {full_path}: {e}")

def save_features(video_paths, directories, video_dir, faces_dir):
    logger.info("Saving features for videos...")
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        executor.map(partial(process_video, directories=directories, video_dir=video_dir, faces_dir=faces_dir), video_paths)

if __name__ == "__main__":
    csv_file = r"data\videos.csv"
    video_list = pd.read_csv(csv_file)["filename"].tolist()
    save_features(video_list, features_dirs, video_dir, faces_dir)