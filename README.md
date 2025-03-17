# Face Recognition

This repository implements a face recognition system that leverages traditional computer vision techniques and machine learning. The project uses:

- **OpenCV** with Haar cascades for face and eye detection.
- **PyWavelets (pywt)** for applying wavelet transforms to extract features.
- **Scikit-Learn's** Support Vector Classifier (SVC) in a pipeline with standard scaling for face classification.
- **Joblib** to save and load the trained model.

The system processes images from a designated folder, crops faces that have at least two detected eyes, extracts features using wavelet transforms, and trains a model to classify faces based on provided training data.

---

## Repository Structure
face_rec/

├── .idea/                   # (IDE configuration files; can be ignored)

├── __pycache__/             # (Python cache files)

├── opencv/haar_cascades/    # Contains pre-trained Haar cascade XML files (e.g., haarcascade_frontalface_default.xml)

├── testimg/                 # Directory containing test images for processing and training

│   └── (subdirectories with images)

├── class_dictionary.json    # JSON file mapping class names to numeric labels

├── saved_model.pkl          # Trained model saved as a pickle file

├── main.py                  # Main script: preprocess images, crop faces, extract features, and train the classifier

├── test.py                  # Script to test face and eye detection on a single image

├── video_run.py             # Script to run face recognition on a video file (e.g., video.mp4)

├── test.mp4                 # (Example video file for testing; optional)

└── video.mp4                # (Example video file used in video_run.py; optional)

---

## Prerequisites

Ensure you have Python 3.7+ installed. The project depends on the following packages:

- [OpenCV](https://opencv.org/) (`opencv-python`)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [PyWavelets](https://pywavelets.readthedocs.io/)
- [Scikit-Learn](https://scikit-learn.org/)
- [Joblib](https://joblib.readthedocs.io/)

You can install the required packages using pip.
```bash
pip install opencv-python numpy matplotlib PyWavelets scikit-learn joblib
```
## Usage

### 1. Training the Model
The `main.py` script performs the following tasks:

- Loads images from the `testimg` folder.
- Detects and crops faces (only if at least two eyes are detected).
- Applies a wavelet transform to each image to extract additional features.
- Resizes and stacks both the raw and wavelet-transformed images.
- Trains a Support Vector Classifier (SVC) using a Scikit-Learn pipeline.
- Saves the trained model to `saved_model.pkl` and the class dictionary to `class_dictionary.json`.

To run the training script, execute:

```bash
python main.py
```
--- 
### 2. Testing Face Detection
The test.py script is a standalone test that:

Reads a sample image from the testimg directory.
Detects faces and eyes using Haar cascades.
Draws rectangles around the detected faces and eyes.
Displays the processed image using Matplotlib.
Run the test script with:

```bash
python test.py
```
---
### 3. Running Face Recognition on Video
The `video_run.py` script demonstrates how to use the saved model for real-time face recognition on a video file:

- Loads the trained model from `saved_model.pkl`.
- Opens and processes a video file (`video.mp4`).
- Detects faces in each frame, applies the wavelet transform, and predicts the face label.
- Draws bounding boxes and labels (e.g., `"CHAEWON"`, `"KAZUHA"`) on the video frames.
- Displays the video with the overlaid predictions.

Run the video recognition script with:

```bash
python video_run.py
```

Note: Adjust the prediction conditions and file paths in the scripts as needed to match your dataset and desired output.

## Customization

### Data Folder
Modify the `path_to_data` variable in `main.py` to point to your image dataset directory.

### Haar Cascades
Ensure the path to the Haar cascade XML files in the `opencv/haar_cascades/` directory is correct.

### Wavelet Parameters
The `w2d` function in `main.py` applies a wavelet transform to extract features. You can adjust the wavelet type (`mode`) and decomposition level as needed.

### Classifier
The pipeline in `main.py` uses an SVC with an RBF kernel. You can experiment with different classifiers or adjust the hyperparameters.

---

## Logging
Training logs are saved in the `logs` folder. You can use TensorBoard to monitor training progress:

```bash
tensorboard --logdir logs
```
After running the command above, open your browser and go to http://localhost:6006/ to view the TensorBoard dashboard.

---
## Acknowledgements

- **OpenCV** for providing robust computer vision tools.
- **PyWavelets** for wavelet transform functionalities.
- **Scikit-Learn** for easy-to-use machine learning pipelines.

For any questions or feedback, feel free to open an issue in the repository.

--- 

## License
This project isn’t licensed. Feel free to use it.
