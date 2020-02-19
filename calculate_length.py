import argparse
import csv
import wave


def parse_length(audio_csv_file):
    total_duration = 0
    with open(audio_csv_file, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            file_path = row['wav_filename']
            fin = wave.open(file_path, 'rb')
            frames = fin.getnframes()
            rate = fin.getframerate()
            duration = frames / float(rate)
            total_duration += duration
            fin.close()
    return total_duration


def main():
    parser = argparse.ArgumentParser(description='Check length of data from DeepSpeech csv file')
    parser.add_argument('--input_path', required=True, type=str,
                        help='Source csv file from DeepSpeech')
    args = parser.parse_args()
    print("Total length:", parse_length(args.input_path), "seconds")


if __name__ == "__main__":
    main()
