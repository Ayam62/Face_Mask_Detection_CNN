# Face Mask Detection

A deep learning project that detects whether a person is wearing a face mask or not using a pre-trained VGG16 model. The project includes a trained Keras model, a FastAPI backend server, and a web-based frontend with real-time webcam integration.

## Features

- **Deep Learning Model**: Fine-tuned VGG16 convolutional neural network
- **High Accuracy**: 93% validation accuracy on test data
- **Real-time Detection**: Live webcam feed with instant predictions
- **REST API**: FastAPI backend for easy integration
- **Web Interface**: HTML/JavaScript frontend with capture functionality
- **CORS Enabled**: Ready for cross-origin requests

## Project Structure

```
Face_Mask_Detection/
├── Face_Mask_Detection.ipynb    # Jupyter notebook with training code
├── face_mask_model.h5           # Trained Keras model
├── server.py                    # FastAPI backend server
├── index.html                   # Web frontend
├── requirements.txt             # Python dependencies
├── data/
│   ├── with_mask/               # Training images (with mask)
│   └── without_mask/            # Training images (without mask)
└── README.md                    # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- Webcam (for testing)

### Setup

1. **Clone or navigate to the project directory**:
   ```bash
   cd Face_Mask_Detection
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Or install individually:
   ```bash
   pip install tensorflow-macos tensorflow-metal keras fastapi uvicorn python-multipart pillow opencv-python numpy scikit-learn
   ```

## Usage

### Training (Jupyter Notebook)

Open and run `Face_Mask_Detection.ipynb` in Jupyter:

```bash
jupyter notebook Face_Mask_Detection.ipynb
```

The notebook:
- Loads images from `data/with_mask/` and `data/without_mask/`
- Preprocesses images (224×224 resize, normalization)
- Trains a VGG16 model on 1,000 sampled images
- Saves the trained model as `face_mask_model.h5`

### Running the Backend Server

Start the FastAPI server:

```bash
uvicorn server:app --reload
```

The server will run at `http://127.0.0.1:8000`

**Available endpoints:**
- `GET /` - Returns API info
- `POST /predict` - Accept image file and return mask detection prediction

**Request example:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@image.jpg"
```

**Response example:**
```json
{
  "prediction": "Mask",
  "confidence": 0.33
}
```

### Running the Frontend

1. Make sure the FastAPI server is running
2. Open `index.html` in a web browser
3. Allow webcam access when prompted
4. Click the **"Capture & Predict"** button to:
   - Capture a frame from your webcam
   - Send it to the backend for inference
   - Display the prediction result

## Model Details

### Architecture

- **Base Model**: VGG16 (pre-trained on ImageNet)
- **Frozen Layers**: All VGG16 layers except the last one
- **Custom Output Layer**: 1 Dense layer with sigmoid activation (binary classification)
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy

### Input Specifications

- **Input Size**: 224 × 224 pixels (RGB)
- **Normalization**: Pixel values scaled to [0, 1]
- **Format**: OpenCV BGR (converted to RGB in preprocessing)

### Output

- **Output Value**: Float between 0 and 1 (sigmoid output)
- **Label 0** (< 0.5): Wearing a mask 
- **Label 1** (≥ 0.5): Not wearing a mask 

## Training Details

- **Dataset**: 1,000 sampled images (balanced mix of with_mask and without_mask)
- **Train/Test Split**: 80/20
- **Epochs**: 5
- **Validation Accuracy**: ~93%

## Technologies Used

- **Deep Learning**: TensorFlow, Keras
- **Backend**: FastAPI, Uvicorn
- **Image Processing**: OpenCV, Pillow, NumPy
- **Frontend**: HTML5, JavaScript
- **Data Processing**: Scikit-learn, Pandas

## License

This project is open source and available for educational and personal use.

## Author

Created as part of the "100 Days of Deep Learning" challenge.

## Acknowledgments

- VGG16 architecture
- FastAPI framework documentation
- OpenCV and TensorFlow communities

---


