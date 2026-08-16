import sounddevice as sd
from preprocessingArray import preprocess_audio3, highpass_filter
import torch
from model import Net
import numpy as np
from collections import deque
import math
import soundfile as sf
import queue

last_audio = None
# Variables the GUI can read
best_prob = 0
second_prob = 0
energy = 0
prediction_text = "Listening..."
second_prediction_text = ""

net = Net()
net.load_state_dict(torch.load("voice_model.pth"))
net.eval()

audio_queue = queue.Queue()

sample_rate = 16000

# ENERGY_THRESHOLD = 0.02
alpha = 0.95

CHUNK_DURATION = 0.05
HANGOVER_TIME = 0.30

hangover_chunks = math.ceil(
    HANGOVER_TIME / CHUNK_DURATION
)

recording = False
silent_chunks = 0

recorded_audio = []

BUFFER_DURATION = 0.5

buffer_size = int(
    BUFFER_DURATION * sample_rate
)

rolling_buffer = deque(
    maxlen=buffer_size
)


classes = {
    0: "Dur",
    1: "Ileri",
    2: "Geri",
    3: "Kalk",
    4: "Unknown"
}


def audio_callback(indata, frames, time, status):
    audio_queue.put(indata.copy())


def start_listening():

    global energy
    global best_prob
    global prediction_text
    global second_prediction_text
    global recording
    global silent_chunks
    global second_prob
    global recorded_audio
    global last_audio
    noise_energy = 0.0

    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=800,
        callback=audio_callback
    )
    stream.start()

    while True:
        audio = audio_queue.get()
        audio = np.squeeze(audio)
        rolling_buffer.extend(audio)

        filtered = highpass_filter(audio, sample_rate, cutoff=80)

        energy = np.sqrt(np.mean(filtered ** 2))
        if not recording:
            noise_energy = alpha * noise_energy + (1 - alpha) * energy

        threshold = max(noise_energy * 3, 0.01)
        if not recording:
            if energy > threshold:
                recording = True
                silent_chunks = 0

                recorded_audio = [
                    np.array(rolling_buffer)
                ]
            continue
        recorded_audio.append(audio)

        if energy > threshold:
            silent_chunks = 0
        else:
            silent_chunks += 1

        if silent_chunks < hangover_chunks:
            continue
        complete_audio = np.concatenate(
            recorded_audio
        )

        last_audio = complete_audio.copy()
        features = preprocess_audio3(
            complete_audio
        )
        duration = len(complete_audio)/ sample_rate

        if features is not None:
            x = torch.tensor(
                features,
                dtype=torch.float32
            )
            x = x.unsqueeze(0)
            x = x.unsqueeze(0)
            with torch.no_grad():
                outputs = net(x)
            probs = torch.softmax(
                outputs,
                dim=1
            )[0]
            top2 = torch.topk(
                probs,
                2
            )
            best_prob = (
                top2.values[0].item() * 100
            )
            second_prob = (
                top2.values[1].item() * 100
            )

            prediction = (
                top2.indices[0].item()
            )

            if (
                best_prob < 85
                or best_prob-second_prob < 20
                or duration > 1.5
            ):
                prediction = 4
            second_prediction = (
                top2.indices[1].item()
            )
            prediction_text = classes[prediction]
            second_prediction_text = classes[second_prediction]

        rolling_buffer.clear()
        recorded_audio = []

        recording = False
        silent_chunks = 0