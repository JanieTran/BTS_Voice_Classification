import librosa
import pandas as pd
import os
import numpy as np
from librosa.feature import mfcc as MFCC
from config import MEMBER_TO_LABEL
from time import time
from scipy import signal


def load_audio(file_path, resample_rate=1.):
    """Load audio file into numpy object, resample if specified
    Args:
        file_path (str): path to audio file
        resample_rate (float): multiplied by original sample rate to get new sample rate,
            <1 for down-sampling, >1 for up-sampling, 1 for no resampling
    Returns:
        np.array: (resampled) audio
        int: (new) sample rate
    """
    x, sample_rate = librosa.load(path=file_path, sr=None)
    if resample_rate == 1:
        return x, sample_rate

    # Calculate new number of samples in resulted signal
    duration = len(x) / sample_rate
    new_sr = sample_rate * resample_rate
    n_sample = int(duration * new_sr)

    audio_resampled = signal.resample(x=x, num=n_sample)
    return audio_resampled, int(new_sr)


def fourier_transform(x, sample_rate, use_window=True):
    """Apply FFT to break down signal to its component frequencies
    Args:
        x (np.array(float)): input signal
        sample_rate (int): sampling rate of signal
        use_window (bool): whether to use Blackman window for FFT
    Returns:
        np.array(float): resulted frequency domain of the same length as input
        np.array(float): frequencies
    """
    n_samples = x.shape[0]
    window = signal.windows.blackmanharris(M=n_samples)

    if use_window:
        fourier = np.fft.fft(x * window)
    else:
        fourier = np.fft.fft(x)

    fourier = fourier[:n_samples // 2]
    fourier = 2 / n_samples * abs(fourier)
    frequencies = np.linspace(start=0, stop=sample_rate / 2, num=n_samples // 2)
    return fourier, frequencies


def get_audio_features_dataframe(n_mfcc):
    """Extract specified audio features from all audio files
    and store as csv
    Args:
        n_mfcc (int): number of MFCC values to extract
    Returns: None
    """
    df = pd.DataFrame()

    for member, label in MEMBER_TO_LABEL.items():
        print(member, end=' ')
        files = os.listdir(f'data/{label}')
        files = [f'data/{label}/{x}' for x in files]
        rows = []

        for f in files:
            # Aggregated MFCC values, so that each audio is represented
            # by a 1D array of length n_mfcc
            x, sample_rate = load_audio(file_path=f, resample_rate=1/3)
            mfcc_feat = MFCC(y=x, sr=sample_rate, n_mfcc=n_mfcc)
            mfcc_mean = mfcc_feat.mean(axis=1)

            # Each coefficient is a feature column
            row = {'label': label, 'file_path': f}
            for number, coef in enumerate(mfcc_mean):
                row[f'mfcc_{n_mfcc}_{number + 1:02d}'] = coef
            rows.append(row)
        df = df.append(rows, ignore_index=True)

    df.to_csv(f'features_mfcc{n_mfcc}_down.csv', index=False)


if __name__ == "__main__":
    t = time()
    get_audio_features_dataframe(n_mfcc=12)
    # Approximately 8 minutes for 1,570 samples of 44100 Hz
    print(f'\nElapsed time: {time() - t:.2f}s')
