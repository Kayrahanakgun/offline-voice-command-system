import numpy as np
import librosa
import noisereduce as nr
from scipy.signal import butter, sosfilt


TARGET_SR = 16000
BIT_DEPTH = 8
MAX_VAL = 2 ** (BIT_DEPTH - 1)
N_FFT = 512
HOP_MS = 10
N_MFCC = 13
TARGET_FRAMES = 221


def pre_emphasize(signal, coefficient=0.97):

    return np.append(
        signal[0],
        signal[1:] - coefficient * signal[:-1]
    )


def highpass_filter(signal, sr, cutoff=50, order=4):

    sos = butter(
        order,
        cutoff,
        btype="highpass",
        fs=sr,
        output="sos"
    )

    return sosfilt(sos, signal)


def lowpass_filter(signal, sr, cutoff=6000, order=4):

    sos = butter(
        order,
        cutoff,
        btype="lowpass",
        fs=sr,
        output="sos"
    )

    return sosfilt(sos, signal)


def pad_or_trim(features, target_frames=TARGET_FRAMES):
    current_frames = features.shape[1]

    if current_frames < target_frames:
        pad_width = target_frames - current_frames
        features = np.pad(
            features,
            ((0, 0), (0, pad_width)),
            mode="constant"
        )
    elif current_frames > target_frames:
        features = features[:, :target_frames]
    return features


def preprocess_audio3(audio, verbose=False):

    sample_rate = TARGET_SR

    audio = np.squeeze(audio)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    audio_quantized = np.int16(audio * MAX_VAL)
    audio_quantized = audio_quantized.astype(np.float32)

    peak = np.max(np.abs(audio_quantized))

    if peak != 0:
        audio_normalized = audio_quantized / peak
    else:
        audio_normalized = audio_quantized

    filter1_audio = highpass_filter(audio_normalized, sample_rate)
    filtered_audio = lowpass_filter(filter1_audio, sample_rate)

    noise = filtered_audio[:int(0.3 * sample_rate)]

    denoised_audio = nr.reduce_noise(
        y=filtered_audio,
        y_noise=noise,
        sr=sample_rate,
        stationary=True,
        prop_decrease=0.8
    )

    peak = np.max(np.abs(denoised_audio))

    if peak != 0:
        denoised_audio = denoised_audio / peak

    pre_emphasized_audio = pre_emphasize(denoised_audio)

    non_silent_intervals = librosa.effects.split(
        pre_emphasized_audio,
        top_db=20,
        frame_length=2000,
        hop_length=512
    )

    if len(non_silent_intervals) > 0:

        speech_only_audio = np.concatenate([
            pre_emphasized_audio[start:end]
            for start, end in non_silent_intervals
        ])

    else:

        speech_only_audio = np.array([], dtype=np.float32)

    if len(speech_only_audio) == 0:
        print("Silence")
        return

    hop_length = int(sample_rate * HOP_MS / 1000)

    mfccs = librosa.feature.mfcc(
        y=speech_only_audio,
        sr=sample_rate,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=hop_length,
        window="hamming"
    )

    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
    final_features = np.concatenate([mfccs, mfcc_delta, mfcc_delta2], axis=0)
    final_features = pad_or_trim(final_features)
    # if verbose:
    #
    #     print(
    #         f"Before silence removal: "
    #         f"{len(pre_emphasized_audio) / sample_rate:.2f} seconds"
    #     )
    #     print(
    #         f"After silence removal:  "
    #         f"{len(speech_only_audio) / sample_rate:.2f} seconds"
    #     )
    #
    #     print(f"Spektrogram Matrisi: {spectrogram_db.shape}")
    #     print(f"MFCC Matrisi (Delta dahil): {final_features.shape}")
    #     print(final_features.shape)

    return final_features


# features = preprocess_audio("data/TestAudio.wav")

# print(features.shape)
