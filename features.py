import librosa
import pandas as pd
import os
import numpy as np
from librosa.feature import mfcc as MFCC
from config import MEMBER_TO_LABEL
from time import time
from scipy import signal


def load_audio(file_path, sample_rate=44100):
    """Load audio file into numpy object, resample if specified
    Args:
        file_path (str): path to audio file
        sample_rate (float): sampling rate to load input
    Returns:
        np.array: (resampled) audio
        int: (new) sample rate
    """
    x, sr = librosa.load(path=file_path, sr=None)
    if sample_rate == sr:
        return x, sample_rate

    # Calculate new number of samples in resulted signal
    duration = len(x) / sr
    n_sample = int(duration * sample_rate)

    audio_resampled = signal.resample(x=x, num=n_sample)
    return audio_resampled, sample_rate


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
        fourier = np.fft.rfft(x * window)
    else:
        fourier = np.fft.rfft(x)

    fourier = 2 / n_samples * abs(fourier)
    frequencies = np.fft.rfftfreq(n=n_samples, d=1/sample_rate)
    return fourier, frequencies


def log_mel_spectrogram(spectrogram, n_mel_bins):
    """Log-power Mel spectrogram
    Args:
        spectrogram (np.array(float)): resulted spectrogram from STFT
        n_mel_bins (int): number of Mel filter banks
    Returns:
        np.array(float): log-power Mel spectrogram
    """
    mel_spec = librosa.feature.melspectrogram(S=np.abs(spectrogram) ** 2, n_mels=n_mel_bins)
    log_mel_spec = librosa.power_to_db(mel_spec)
    return log_mel_spec


def mel_freq_cepstral_coeff(spectrogram, n_mel_bins, n_mfcc):
    """Mel Frequency Cepstral Coefficients
    Args:
        spectrogram (np.array(float)): resulted spectrogram from STFT
        n_mel_bins (int): number of Mel filter banks
        n_mfcc (int): number of resulted coefficients
    Returns:
        np.array(float): MFCC of specified coefficients
    """
    log_mel_spec = log_mel_spectrogram(spectrogram=spectrogram, n_mel_bins=n_mel_bins)
    mfccs = librosa.feature.mfcc(S=log_mel_spec, n_mfcc=n_mfcc)
    return mfccs


def spectral_centroid(spectrogram, sample_rate):
    """Weighted average of frequencies
    Args:
        spectrogram (np.array(float)): resulted spectrogram from STFT
        sample_rate (int): sampling rate of signal
    Returns:
        np.array(float): spectral centroid
    """
    X = np.abs(spectrogram) ** 2
    norm = X.sum(axis=0, keepdims=True)
    norm[norm == 0] = 1

    centroids = np.dot(np.arange(0, X.shape[0]), X) / norm
    centroids = centroids / (X.shape[0] - 1) * sample_rate / 2

    return np.squeeze(centroids)


def spectral_flux(spectrogram):
    """Difference in frequency of successive time frames
    Args:
        spectrogram (np.array(float)): resulted spectrogram from STFT
    Returns:
        np.array(float): spectral flux
        np.array(float): timestamps
    """
    # Squared difference in frequency between successive frames
    flux = np.c_[spectrogram[:, 0], spectrogram]
    flux = np.abs(np.diff(flux, n=1, axis=1))
    flux = np.sqrt(np.sum(flux ** 2, axis=0)) / spectrogram.shape[0]
    return flux


def spectral_slope(spectrogram):
    # Mean
    mu = spectrogram.mean(axis=0, keepdims=True)
    # Index vector
    kmu = np.arange(0, spectrogram.shape[0]) - spectrogram.shape[0] / 2
    # Slope
    slope = spectrogram - mu
    slope = np.dot(kmu, slope) / np.dot(kmu, kmu)
    return slope


def spectral_roll_off(spectrogram, sample_rate, cutoff=0.85):
    # Sum of frequency energies to calculate mean
    freq_sum = spectrogram.sum(axis=0)
    freq_sum[freq_sum == 0] = 1

    # Cumulative sum of energy for each frequency
    X = np.cumsum(spectrogram, axis=0)
    # Divide by total energy sum to find much energy is covered up to a frequency
    X = X / freq_sum

    # Find position of frequency covering cutoff percentage
    roll_off = np.argmax(X >= cutoff, axis=0)
    # Convert from position index to Hz
    roll_off = roll_off / (X.shape[0] - 1) * sample_rate / 2
    return roll_off


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
            x, sample_rate = load_audio(file_path=f, sample_rate=44100/3)
            mfcc_feat = MFCC(y=x, sr=sample_rate, n_mfcc=n_mfcc)
            mfcc_mean = mfcc_feat.mean(axis=1)
            zero_crossing = sum(librosa.zero_crossings(y=x))
            centroids = librosa.feature.spectral_centroid(y=x, sr=sample_rate)

            # Each coefficient is a feature column
            row = {'label': label, 'file_path': f}
            for number, coef in enumerate(mfcc_mean):
                row[f'mfcc_{n_mfcc}_{number + 1:02d}'] = coef
            row['zero_crossing'] = zero_crossing
            row['spectral_centroid'] = np.mean(centroids)
            rows.append(row)
        df = df.append(rows, ignore_index=True)

    df.to_csv(f'features.csv', index=False)


if __name__ == "__main__":
    t = time()
    get_audio_features_dataframe(n_mfcc=12)
    # Approximately 8 minutes for 1,570 samples of 44100 Hz
    print(f'\nElapsed time: {time() - t:.2f}s')
