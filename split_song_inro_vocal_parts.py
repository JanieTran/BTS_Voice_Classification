import json
import os
from pydub import AudioSegment
from config import *

with open('audio/label.json') as f:
    ANNOTATION = json.load(f)


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


def main():
    # mp3_files = [x.split('.')[0] for x in os.listdir('audio') if '.mp3' in x]
    mp3_files = ['DNA']

    for song_name in mp3_files:
        print(song_name, end=' ')
        song = AudioSegment.from_file(f'audio/{song_name}.mp3')
        count = 0

        for timestamp, member in ANNOTATION[song_name].items():
            start_ms, end_ms = get_start_end_ms(timestamp=timestamp)
            # If track is longer than 5s, split into smaller tracks
            if end_ms - start_ms > MAX_LEN_MS:
                cursor_start = start_ms
                while cursor_start < end_ms:
                    cursor_end = min(end_ms, cursor_start + MAX_LEN_MS)
                    sub_track = song[cursor_start:cursor_end]
                    # Only export tracks longer than min length
                    if len(sub_track) >= MIN_LEN_MS:
                        sub_track.export(f'data/{MEMBER_LABELS[member]}/{song_name}_{count}.mp3', format='mp3')
                    cursor_start = cursor_end
                    count += 1
            # Only export tracks longer than min length
            elif end_ms - start_ms >= MIN_LEN_MS:
                track = song[start_ms:end_ms]
                track.export(f'data/{MEMBER_LABELS[member]}/{song_name}_{count}.mp3', format='mp3')
                count += 1

    print()
    total = 0
    for i in range(7):
        count = len(os.listdir(f'data/{i}'))
        print(f'{i}: {count}')
        total += count
    print(f'Total: {total}')


if __name__ == '__main__':
    main()
