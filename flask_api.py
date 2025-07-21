import os
import base64
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import logging
import traceback

from backend_processing import analyze_video, analyze_image

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'temp_uploads'
TRANSFORMED_FOLDER = 'transformed_videos'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRANSFORMED_FOLDER, exist_ok=True)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def image_to_base64(path):
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Could not read and encode image {path}: {e}")
        return None

@app.route('/analyze', methods=['POST'])
def handle_analysis():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    input_type = request.form.get('type', 'Image')

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    files_to_delete = [filepath]
    transformed_video_path = None
    gradcam_paths = []
    feature_threshold = None

    try:
        if input_type == 'Video':
            logger.info(f"Initiating VIDEO analysis for: {file.filename}")
            prediction, confidence, frame_metrics, feature_metrics, transformed_video_path, gradcam_paths, feature_threshold = analyze_video(filepath)
            
        else: # Image
            logger.info(f"Initiating IMAGE analysis for: {file.filename}")
            prediction, confidence, frame_metrics, feature_metrics, gradcam_path, feature_threshold = analyze_image(filepath)
            gradcam_paths = [gradcam_path] if gradcam_path else []

        files_to_delete.extend(g for g in gradcam_paths if g)
        
        gradcam_images_base64 = [image_to_base64(p) for p in gradcam_paths if p]
        
        summary_stats = {}
        if frame_metrics:
            metrics_np = np.array(frame_metrics)
            summary_stats = {
                'average': float(np.mean(metrics_np)),
                'highest': float(np.max(metrics_np)),
                'lowest': float(np.min(metrics_np))
            }
        else:
            summary_stats = {'average': 0.0, 'highest': 0.0, 'lowest': 0.0}

        # The path sent to the frontend should be a URL path, not a file system path
        video_url_path = None
        if transformed_video_path:
            video_filename = os.path.basename(transformed_video_path)
            video_url_path = f'transformed_videos/{video_filename}'


        response_data = {
            'prediction': prediction,
            'confidence': confidence,
            'frameMetrics': frame_metrics,
            'gradcamImages': gradcam_images_base64,
            'summaryStats': summary_stats,
            'featureMetrics': feature_metrics,
            'transformedVideoPath': video_url_path,
            'featureThreshold': feature_threshold
        }
        
        logger.info(f"Analysis complete for {file.filename}. Prediction: {prediction}")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error during analysis: {e}\n{traceback.format_exc()}")
        return jsonify({'error': f'Server error during analysis: {str(e)}'}), 500
    finally:
        for temp_file in files_to_delete:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {temp_file}: {e}")

@app.route('/transformed_videos/<path:filename>')
def serve_transformed_video(filename):
    logger.info(f"Serving video for inline playback: {filename}")
    try:
        return send_from_directory(
            TRANSFORMED_FOLDER,
            filename,
            as_attachment=False,
            mimetype='video/mp4'
        )
    except FileNotFoundError:
        logger.error(f"File not found: {filename} in {TRANSFORMED_FOLDER}")
        return jsonify({'error': 'The requested video file was not found on the server.'}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)