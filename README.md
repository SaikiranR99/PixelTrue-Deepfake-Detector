<<<<<<< HEAD
# PixelTrue: AI-Powered Deepfake Detection System

PixelTrue is a full-stack web application designed for the advanced detection of deepfakes in both images and videos. It leverages a powerful AI model (Xception) and provides a detailed, user-friendly dashboard to visualize the analysis results.

## ✨ Features

- **Dual-Mode Analysis:** Supports both **image** and **video** file uploads for deepfake detection.
- **AI-Powered Prediction:** Utilizes fine-tuned Xception models to provide a clear prediction ("Deepfake" or "Real").
- **Confidence Scoring:** Displays a percentage confidence score for each analysis.
- **In-Depth Video Metrics:** For videos, it offers frame-by-frame scoring, summary statistics (average, max, min scores), and an interactive score density chart.
- **Visual Evidence (Grad-CAM):** Generates and displays Grad-CAM (Gradient-weighted Class Activation Mapping) heatmaps, highlighting the specific regions of an image or frame that the model focused on to make its prediction.
- **Feature Profile:** For images, it plots a feature metrics profile, offering a deeper look into the data points used for classification.
- **Interactive Dashboard:** A sleek, futuristic user interface built with Next.js and Recharts for dynamic data visualization.
- **Annotated Video Output:** For detected deepfake videos, it generates and provides a link to an annotated version of the video with frame-by-frame probability scores overlaid.

---

## 🎥 Demo

Watch a brief video demonstration of the PixelTrue dashboard in action. Click the thumbnail below to see how to upload a file and interpret the analysis results.

[![PixelTrue Demo Video](https://img.youtube.com/vi/oHXcZ487PSk/maxresdefault.jpg)](https://www.youtube.com/watch?v=oHXcZ487PSk)

---

## 🏗️ Architecture

The application is built with a decoupled frontend and backend:

-   **Backend (Flask):** A Python-based API server that handles file uploads, performs the heavy lifting of AI analysis using TensorFlow/Keras, and serves the results and processed files.
-   **Frontend (Next.js):** A modern, responsive web application built with React/Next.js and styled with Tailwind CSS. It provides the user interface for file submission and presents the complex analysis data in an intuitive, interactive dashboard.

---

## ⚙️ Prerequisites

Before you begin, ensure you have the following installed:

-   **Python 3.10.6**
-   **Node.js** (v18.x or later recommended) and **npm**
-   A Python virtual environment tool like `venv`

---
## 💾 Datasets
The models were trained on a variety of datasets. You can access the specific datasets used for training and evaluation via the following link:

[Deepfake Detection Datasets (Google Drive)](https://drive.google.com/drive/u/0/folders/1R_aPPGFf6y1f26Z7FheUFCxN6h1bxbCY)

## ⚙️ Prerequisites

Before you begin, ensure you have the following installed:

-   **Python 3.10.6**
-   **Node.js** (v18.x or later recommended) and **npm**
-   A Python virtual environment tool like `venv`


## 🚀 Installation & Setup

Follow these steps to get the application running locally.

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>

2. Backend Setup (Flask API)
The backend requires several Python packages for machine learning, image/video processing, and serving the API.

a. Create and Activate Virtual Environment:
From the project root directory:

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate

b. Install Python Dependencies:
Create a file named requirements.txt in the root directory and paste the content below into it.

requirements.txt

tensorflow
numpy
pandas
scikit-learn
matplotlib
seaborn
opencv-python
mtcnn
PyWavelets
torch
torchvision
tqdm
scenedetect
Flask
Flask-Cors

Now, install these packages using pip:

pip install -r requirements.txt

c. Place Pre-trained Models:
Ensure your pre-trained model files are located in a models folder in the root directory. The application expects to find:

models/Xception_image_model.h5

models/Xception_model.h5

3. Frontend Setup (Next.js Dashboard)
The frontend is located in the dashboard directory.

a. Navigate to the Frontend Directory:

cd dashboard

b. Install Node.js Dependencies:

npm install

▶️ How to Run the Application
The application requires both the backend and frontend servers to be running simultaneously.

1. Start the Backend API Server:
Open a terminal, navigate to the root directory of the project, and activate your virtual environment. Then run:

python flask_api.py

The Flask API will start running on http://localhost:5000.

2. Start the Frontend Development Server:
Open a second terminal, navigate to the dashboard directory, and run:

npm run dev

The Next.js application will start and be accessible at http://localhost:3000.

👤 How to Use
Open your web browser and navigate to http://localhost:3000.

You will be on the landing page. Click the Launch Dashboard button.

You will be redirected to the login page. Use the following credentials to log in:

Email: rangarajans@cardiff.ac.uk

Password: pixeltrue

After logging in, you will be on the main analysis dashboard.

Use the System Control panel on the left to:

Select the analysis type (Image or Video).

Select a file from your local machine.

Click Initiate Scan to begin the analysis.

The dashboard will update in real-time to show the analysis progress and display the final results upon completion.

📁 Project Structure
.
├── dashboard/              # Next.js frontend application
│   ├── app/
│   │   ├── dashboard/page.tsx  # Main dashboard component
│   │   ├── login/page.tsx      # Login page component
│   │   └── page.tsx            # Landing page component
│   └── ...
├── models/                 # Folder for trained .h5 models
│   ├── Xception_image_model.h5
│   └── Xception_model.h5
├── temp_uploads/           # Temp folder for file uploads (auto-created)
├── transformed_videos/     # Folder for annotated videos (auto-created)
├── venv/                   # Python virtual environment
├── backend_processing.py   # Core analysis logic for the API
├── flask_api.py            # Main Flask API file
├── preprocessing_images.py # Scripts for data preprocessing
├── preprocessing_videos.py #
├── train_images.py         # Scripts for model training
├── train_videos.py         #
└── requirements.txt        # Python dependencies
=======
# PixelTrue---A-Deepfake-Detection-Application
PixelTrue is a sophisticated, full-stack web application built to identify deepfakes in both images and videos. At its core, it uses a powerful AI model to analyze media files and determine if they are authentic or have been synthetically manipulated.
>>>>>>> 69296c2c637924daac9ad8ffa3bee91167085004
