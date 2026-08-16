import os
from preprocessingArray import preprocess_audio3
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import librosa

TARGET_SR = 16000


def load_dataset():

    X = []
    y = []

    classes = {
        "Dur": 0,
        "Ileri": 1,
        "Geri": 2,
        "Kalk": 3,
        "Unknown": 4
    }

    for class_name, label in classes.items():

        folder = os.path.join("data", class_name)

        for filename in os.listdir(folder):

            if filename.endswith(".wav"):

                filepath = os.path.join(folder, filename)

                audio, sr = librosa.load(
                    filepath,
                    sr=TARGET_SR,
                    mono=True
                )

                features = preprocess_audio3(audio)

                X.append(features)
                y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(y, dtype=torch.long)

    return X, y

X, y = load_dataset()
dataset = TensorDataset(X, y)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

for inputs, labels in loader:
    print(inputs.shape)
    print(labels)