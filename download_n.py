import argparse
import csv
import os
import gzip
import shutil
import wave
import wget
import xml.etree.ElementTree as ET


def parse_folder(source_path, target_path, target_duration):
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    subfolders = [os.path.join(source_path, subfolder) for subfolder in os.listdir(source_path)
                  if os.path.isdir(os.path.join(source_path, subfolder))]
    files_to_download = []
    for subfolder in subfolders:
        data_list = os.listdir(subfolder)
        for file_name in data_list:
            if '.xml.gz' in file_name:
                files_to_download.append(os.path.join(subfolder, file_name))
    total_length = 0
    while total_length < target_duration:
        for file_to_download in files_to_download:
            total_length += parse_file(file_to_download, target_path)
            if total_length >= target_duration:
                break
    print("Finished download. Downloaded", total_length, "seconds of audio in folder:", target_path)


def parse_file(file_path, target_path):
    tree = ET.parse(gzip.open(file_path, 'r'))
    root = tree.getroot()

    file_name, extension = os.path.splitext(file_path)
    cwd = os.getcwd()
    os.chdir(target_path)

    # download
    url = root.find('call-data').find('url').text
    url = url.replace("127.0.0.1:80/convert.sh", "media.yoummday.com")
    download_filename = url.split('/')[-1]
    if not os.path.exists(download_filename):
        wget.download(url)

    # split data
    total_duration = 0
    channels = root.findall('annotation')
    for channel_index, channel in enumerate(channels):
        channel_string = ''
        if len(channels) > 1:
            channel_string = ' remix ' + str(channel_index + 1)
        output_filename = download_filename.rstrip(".mp3") + '_' + str(channel_index + 1) + '.wav'
        os.system("sox " + download_filename + ' --bits 16 -V1 -c 1 --no-dither --encoding signed-integer --endian little '
                                               '--compression 0.0 ' + output_filename + channel_string)
        fin = wave.open(output_filename, 'rb')
        frames = fin.getnframes()
        rate = fin.getframerate()
        duration = frames / float(rate)
        total_duration += duration
        fin.close()
    os.remove(download_filename)
    os.chdir(cwd)
    return total_duration


def create_csv_file(target_path):
    files = os.listdir(target_path)
    filenames = []
    for file in files:
        if '.wav' in file:
            filenames.append(file)
    with open(os.path.join(target_path, 'files.csv'), 'w', newline='') as csvfile:
        fieldnames = ['filename']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for filename in filenames:
            writer.writerow({'filename': os.path.join(target_path, filename)})


def main():
    parser = argparse.ArgumentParser(description='Download eml files of given duration')
    parser.add_argument('--input_path', required=True, type=str,
                        help='Source folder with EML XML files')
    parser.add_argument('--target_path', required=True, type=str,
                        help='Target for download')
    parser.add_argument('--target_duration', required=True, type=int,
                        help='Given target duration in secs')
    args = parser.parse_args()
    if os.path.exists(args.target_path):
        if os.listdir(args.target_path):
            print("Target folder not empty. Should I remove? [y/n]")
            remove = input()
            while remove != 'y' and remove != 'n':
                print("Wrong input. Target folder not empty. Should I remove? [y/n]")
                remove = input()
            if remove == 'y':
                shutil.rmtree(args.target_path)
            else:
                print("Exiting.")
                exit(0)
    print("Downloading files.")
    parse_folder(args.input_path, args.target_path, args.target_duration)
    create_csv_file(args.target_path)


if __name__ == "__main__":
    main()
