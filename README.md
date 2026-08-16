Offline Voice Command Recognition System

An offline, real-time voice command recognition system developed in Python using machine learning and digital signal processing techniques.

The system continuously monitors microphone input and detects when a user speaks. Once speech is detected, the audio is processed and classified by a Convolutional Neural Network (CNN) to identify the predefined command.

The current implementation is designed to recognize four predefined voice commands, but the system can be extended to support additional commands by expanding the dataset and retraining the model.

The entire recognition pipeline operates offline, without requiring an internet connection or cloud-based speech recognition service.

System Requirements
Operating System: Windows 10/11 (64-bit)
Python: 3.11.x
PyTorch: 2.7.1
Microphone: Required for real-time recognition
Dependencies

Install the required Python packages using:

pip install torch torchvision torchaudio
pip install librosa
pip install numpy
pip install scipy
pip install noisereduce
pip install sounddevice
pip install soundfile
pip install matplotlib
pip install customtkinter

Alternatively, if a requirements.txt file is provided:

pip install -r requirements.txt
Project Structure
project/
│
├── train.py
├── model.py
├── dataset.py
├── preprocessingArray.py
├── predictListen.py
├── main.py
├── voice_model.pth
└── data/
File Descriptions
File	Description
train.py	Trains the CNN using the processed voice-command dataset.
model.py	Defines the CNN architecture used for command classification.
dataset.py	Handles loading and preparation of the training dataset.
preprocessingArray.py	Performs audio preprocessing and feature extraction.
predictListen.py	Handles microphone input and real-time command prediction.
main.py	Provides the graphical user interface for the recognition system.
voice_model.pth	Stores the trained PyTorch model parameters.
data/	Contains the audio dataset used for training and evaluation.
Audio Processing Pipeline

Raw microphone audio passes through several preprocessing stages before being classified by the neural network.

Raw Audio
    ↓
8-bit Quantization
    ↓
Normalization
    ↓
Pre-emphasis
    ↓
High-pass Filter
    ↓
Low-pass Filter
    ↓
Noise Reduction
    ↓
Silence Removal
    ↓
MFCC Extraction
    ↓
CNN Classification
    ↓
Predicted Command
1. Audio Acquisition

Audio is captured from the microphone at a sampling rate of 16 kHz.

2. 8-bit Quantization

The recorded signal is quantized to simulate a lower-precision audio representation and support future development toward resource-constrained embedded systems.

3. Normalization

The audio amplitude is normalized to maintain a consistent signal range.

4. Pre-emphasis

A pre-emphasis filter is applied to strengthen higher-frequency components of the speech signal.

5. High-pass and Low-pass Filtering

Frequency components outside the useful speech range are reduced using digital filtering.

6. Noise Reduction

Noise-reduction processing is applied to reduce unwanted background noise.

7. Silence Removal

Non-speech sections are removed so that the model primarily processes the spoken command.

8. MFCC Extraction

Mel-Frequency Cepstral Coefficients (MFCCs) are extracted from the processed audio signal and used as input features for the CNN.

The system uses 13 MFCC coefficients.

CNN Architecture

The extracted MFCC features are classified using a Convolutional Neural Network implemented with PyTorch.

Input
  ↓
Conv2D
  ↓
ReLU
  ↓
MaxPool
  ↓
Conv2D
  ↓
ReLU
  ↓
MaxPool
  ↓
Flatten
  ↓
Linear
  ↓
Dropout
  ↓
Linear
  ↓
Softmax
  ↓
Predicted Command

The convolutional layers learn patterns within the MFCC representation, while the fully connected layers perform the final command classification.

Training Parameters
Parameter	Value
Sample Rate	16,000 Hz
FFT Size	512
MFCC Coefficients	13
Batch Size	16
Optimizer	Adam
Learning Rate	0.001
Weight Decay	1e-4
Scheduler	ReduceLROnPlateau
Epochs	100
Real-Time Recognition

During operation, the system continuously monitors microphone input.

When speech is detected:

The audio segment is captured.
The preprocessing pipeline is applied.
MFCC features are extracted.
The features are passed to the trained CNN.
The network calculates classification scores for the predefined commands.
The command with the highest prediction score is selected.
The result is displayed through the application interface.

Because processing and inference are performed locally, the system does not require an internet connection.

Running the Project

First, install all required dependencies.

Train the Model
python train.py
Run Real-Time Voice Recognition
python predictListen.py
Launch the GUI
python main.py

Note: The exact execution sequence may depend on the project configuration and the location of the trained model and dataset.

Limitations
Recognition performance may decrease in crowded or noisy environments.
Similar-sounding commands may occasionally be confused.
The current system is optimized for short predefined command words.
Recognition performance depends on the diversity and quality of the training dataset.
The system is designed for predefined commands rather than unrestricted speech recognition.
Future Improvements
Quantize and optimize the CNN for deployment on microcontrollers and embedded systems.
Increase the size and diversity of the training dataset.
Support multiple languages.
Add wake-word detection.
Improve robustness against background conversations and environmental noise.
Expand the number of supported predefined commands.
Evaluate lightweight neural-network architectures for low-power hardware.
Technologies Used
Python
PyTorch
NumPy
SciPy
Librosa
SoundDevice
SoundFile
NoiseReduce
Matplotlib
CustomTkinter
Digital Signal Processing
Convolutional Neural Networks
MFCC Feature Extraction
License

This project is released as open-source software under the license included in this repository.
