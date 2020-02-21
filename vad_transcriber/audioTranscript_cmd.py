import shutil
import sys
import os
import logging
import argparse
import subprocess
import shlex
import numpy as np
import wavTranscriber
import wave

# Debug helpers
from utils import parse_clean_eml_file, download_audio

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


def main(args):
    parser = argparse.ArgumentParser(description='Transcribe long audio files using webRTC VAD')
    parser.add_argument('--aggressive', type=int, choices=range(4), required=False,
                        help='Determines how aggressive filtering out non-speech is. (Interger between 0-3)')
    parser.add_argument('--audio', required=False,
                        help='Path to the audio file to run (WAV format)')
    parser.add_argument('--model', required=True,
                        help='Path to directory that contains all model files (output_graph, lm and trie)')
    parser.add_argument('--stream', required=False, action='store_true',
                        help='To use deepspeech streaming interface')
    parser.add_argument('--lm', required=False,
                        help='Path to the language model binary file')
    parser.add_argument('--trie', required=False,
                        help='Path to the language model trie file created with native_client/generate_trie')
    parser.add_argument('--normalize', dest='normalize', action='store_true')
    parser.add_argument('--no_normalize', dest='normalize', action='store_false')
    parser.add_argument('--save_segments', action='store_true')
    parser.add_argument('--save_without_pause', '--save_vad', action='store_true')
    parser.add_argument('--eml_file', type=str)
    parser.set_defaults(normalize=True)
    args = parser.parse_args()
    # Point to a path containing the pre-trained models & resolve ~ if used
    dirName = os.path.expanduser(args.model)

    # Resolve all the paths of model files
    if args.lm or args.trie:
        lm = args.lm
        print("LM:", lm)
        trie = args.trie
        print("trie:", trie)
        output_graph = args.model
        print("model:", output_graph)
    else:
        output_graph, lm, trie = wavTranscriber.resolve_models(dirName)

    # Load output_graph, alpahbet, lm and trie
    model_retval = wavTranscriber.load_model(output_graph, lm, trie)

    if args.eml_file is not None:
        if os.path.isdir(args.eml_file):
            print("eml_file is directory, infering XML file.")
            file_list = os.listdir(args.eml_file)
            for file in file_list:
                if '.xml' in file:
                    args.eml_file = os.path.join(args.eml_file, file)
                    print('using XML file:', args.eml_file)

    if args.audio is None and args.eml_file is not None:
        args.audio = download_audio(args.eml_file)

    if args.audio is None:
        print("Provide argument audio or eml_file in argument")
        exit(1)

    if args.eml_file:
        clean_eml = parse_clean_eml_file(args.eml_file)
        with open(args.audio.replace(".wav", "_EML.txt"), 'w') as eml_file:
            for text in clean_eml:
                eml_file.write(text)
                eml_file.write('\n')

    title_names = ['Filename', 'Duration(s)', 'Inference Time(s)', 'Model Load Time(s)', 'LM Load Time(s)']
    print("\n%-30s %-20s %-20s %-20s %s" % (title_names[0], title_names[1], title_names[2], title_names[3], title_names[4]))

    inference_time = 0.0

    # Run VAD on the input file
    waveFile = args.audio
    segments, sample_rate, audio_length = wavTranscriber.vad_segment_generator(waveFile, args.aggressive,
                                                                               model_sample_rate=model_retval[3],
                                                                               normalize=args.normalize)

    segments = list(segments)

    if args.save_segments:
        if os.path.isdir("segments"):
            shutil.rmtree("segments")
        os.mkdir("segments")

    if args.save_without_pause:
        whole_audio = b''.join(segments)
        new_audio_file = waveFile.rstrip(".wav") + "_VAD.wav"
        with wave.open(new_audio_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(model_retval[3])
            wf.writeframes(whole_audio)

    # save transcript
    with open(waveFile.rstrip(".wav") + "_DeepSpeech.txt", 'w') as f:
        logging.debug("Saving Transcript @: %s" % waveFile.rstrip(".wav") + ".txt")
        for i, segment in enumerate(segments):
            # Run deepspeech on the chunk that just completed VAD
            logging.debug("Processing chunk %002d" % (i,))
            if args.save_segments:
                with wave.open("chunks/chunk{}.wav".format(i), 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(model_retval[3])
                    wf.writeframes(segment)
            audio = np.frombuffer(segment, dtype=np.int16)
            output = wavTranscriber.stt(model_retval[0], audio, sample_rate)
            if not output[0] or output[0] == ' ':
                continue
            inference_time += output[1]
            logging.debug("Transcript: %s" % output[0])

            f.write(output[0] + "\n")

    # Extract filename from the full file path
    filename, ext = os.path.split(os.path.basename(waveFile))
    logging.debug("************************************************************************************************************")
    logging.debug("%-30s %-20s %-20s %-20s %s" % (title_names[0], title_names[1], title_names[2], title_names[3], title_names[4]))
    logging.debug("%-30s %-20.3f %-20.3f %-20.3f %-0.3f" % (filename + ext, audio_length, inference_time, model_retval[1], model_retval[2]))
    logging.debug("************************************************************************************************************")
    print("%-30s %-20.3f %-20.3f %-20.3f %-0.3f" % (filename + ext, audio_length, inference_time, model_retval[1], model_retval[2]))


if __name__ == '__main__':
    main(sys.argv[1:])
