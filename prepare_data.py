import os
import random
import numpy as np
from pydub import AudioSegment
from config import *
from features import load_audio, mel_freq_cepstral_coeff
from scipy import signal


def get_start_end_ms(timestamp):
    """
    Get start and end stamp in millisecond within the song duration
    :param timestamp: start-end string, e.g. '00:00-00:20'
    :return: tuple of start and end stamp in millisecond
    """
    start = list(map(int, timestamp.split('-')[0].split(':')))
    end = list(map(int, timestamp.split('-')[1].split(':')))

    start_ms = start[0] * 60 * 1000 + start[1] * 1000
    end_ms = end[0] * 60 * 1000 + end[1] * 1000

    return start_ms, end_ms


def sample_train_test_songs(test_size, random_seed, include_val=False):
    songs = {'group': [], 'unit': [], 'solo': []}

    for song, attr in ANNOTATION.items():
        song_type = ANNOTATION[song]['type']
        songs[song_type].append(song)

    n_test = int(len(songs['group']) * test_size)
    random.seed(a=random_seed)
    test_songs = random.choices(songs['group'], k=n_test)
    train_songs = [x for x in songs['group'] if x not in test_songs] + songs['unit'] + songs['solo']
    val_songs = None

    if include_val:
        val_songs = test_songs[:n_test // 2]
        test_songs = test_songs[n_test // 2:]
    return train_songs, val_songs, test_songs


def form_audio_data_array(song_names, sample_rate=SAMPLE_RATE, window=WINDOW, n_fft=N_FFT,
                          hop_length=HOP_LENGTH, n_mel=N_MEL_BINS, n_mfcc=N_MFCC, add_axis=False):
    X = []
    Y = []

    for member, label in MEMBER_TO_LABEL.items():
        print(member, end=' ')
        files = os.listdir(f'data/{label}')
        files = [x for x in files if x.split('_')[0] in song_names]
        files = [f'data/{label}/{x}' for x in files]
        for f in files:
            x, sr = load_audio(file_path=f, sample_rate=sample_rate)
            if len(x) / sr < 5:
                continue
            freqs, ts, spectrogram = signal.stft(x=x, fs=sr, window=window, nfft=n_fft, nperseg=hop_length)
            mfccs = mel_freq_cepstral_coeff(spectrogram=spectrogram, n_mel_bins=n_mel, n_mfcc=n_mfcc)
            X.append(mfccs)
            Y.append(label)

    print()
    X = np.array(X)
    Y = np.array(Y)
    if add_axis:
        X = X[..., np.newaxis]
        Y = Y[..., np.newaxis]
    return X, Y


def split_audio_files_into_tracks():
    # mp3_files = [x.split('.')[0] for x in os.listdir('audio') if '.mp3' in x]
    mp3_files = ['Moonchild']

    for song_name in mp3_files:
        print(song_name, end=' ')
        song = AudioSegment.from_file(f'audio/{song_name}.mp3')
        count = 0

        for timestamp, member in ANNOTATION[song_name]['timestamp'].items():
            start_ms, end_ms = get_start_end_ms(timestamp=timestamp)
            # If track is longer than 5s, split into smaller tracks
            if end_ms - start_ms > MAX_LEN_MS:
                cursor_start = start_ms
                while cursor_start < end_ms:
                    cursor_end = min(end_ms, cursor_start + MAX_LEN_MS)
                    sub_track = song[cursor_start:cursor_end]
                    # Only export tracks longer than min length
                    if len(sub_track) >= MIN_LEN_MS:
                        sub_track.export(f'data/{MEMBER_TO_LABEL[member]}/{song_name}_{count}.mp3', format='mp3')
                    cursor_start = cursor_end
                    count += 1
            # Only export tracks longer than min length
            elif end_ms - start_ms >= MIN_LEN_MS:
                track = song[start_ms:end_ms]
                track.export(f'data/{MEMBER_TO_LABEL[member]}/{song_name}_{count}.mp3', format='mp3')
                count += 1

    print()
    total = 0
    for i in range(7):
        count = len(os.listdir(f'data/{i}'))
        print(f'{LABEL_TO_MEMBER[i]}: {count}')
        total += count
    print(f'Total: {total}')


if __name__ == '__main__':
    split_audio_files_into_tracks()
