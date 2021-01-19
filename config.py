import json


MEMBER_TO_LABEL = {
    'Jin': 0, 'Suga': 1, 'J-Hope': 2, 'RM': 3, 'Jimin': 4, 'V': 5, 'Jungkook': 6
}
LABEL_TO_MEMBER = {label: member for member, label in MEMBER_TO_LABEL.items()}

MAX_LEN_MS = 5000
MIN_LEN_MS = 2000

NUM_LABELS = len(MEMBER_TO_LABEL)

with open('data/label.json') as f:
    ANNOTATION = json.load(f)

# Vocal range of singing voice, from E2 to C6
MIN_FREQ = 82
MAX_FREQ = 1047

SAMPLE_RATE = 44100 / 3

N_FFT = 2048
HOP_LENGTH = 512
N_MEL_BINS = 128
N_MFCC = 12

WINDOW = 'blackmanharris'