import os
import shlex
import xml.etree.ElementTree as ET

import wget


def remove_eml_noise_text(text):
    noise_texts = ['~SIL', '~NOISE', '~HESI', '~NonSpeech']
    for noise_text in noise_texts:
        text = text.replace(noise_text, '')
    removed = True
    while removed:
        if '  ' in text:
            text = text.replace('  ', ' ')
        else:
            removed = False
    return text.strip()


def parse_clean_eml_file(file_path, read_channels=1):
    tree = ET.parse(file_path)
    root = tree.getroot()

    texts = []
    channels = root.findall('annotation')
    for channel_index, channel in enumerate(channels):
        if channel_index + 1 > read_channels:
            continue
        this_text = []
        annotations = channel.find("type")
        for item_index, item in enumerate(annotations):
            if item.attrib['type'] == "text":
                text = item.text
                clean_text = remove_eml_noise_text(text).replace('_dot', 'punkt').replace('_at', 'at')
                if not clean_text:
                    continue
                this_text.append(clean_text)
        texts.append('\n'.join(this_text))
    return texts


def download_audio(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    url = root.find('call-data').find('url').text
    url = url.replace("127.0.0.1:80/convert.sh", "media.yoummday.com")
    head, tail = os.path.split(file_path)
    output_dir = head
    output_path = os.path.join(head, tail.replace(".xml", ".wav"))
    if not os.path.exists(output_path):
        print("Downloading missing audio file.")
        filename = wget.download(url, out=output_dir)
        os.system("sox " + shlex.quote(filename) + " " + shlex.quote(output_path))
        os.remove(filename)
        print("Audio file saved in:", shlex.quote(output_path))
    return output_path
