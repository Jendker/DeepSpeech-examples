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
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


def main(args):
    parser = argparse.ArgumentParser(description='Transcribe long audio files using webRTC VAD or use the streaming interface')
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
    parser.add_argument('--save_without_pause', action='store_true')
    parser.set_defaults(normalize=True)
    args = parser.parse_args()
    if args.stream is True:
        print("Opening mic for streaming")
    elif args.audio is not None:
        logging.debug("Transcribing audio file @ %s" % args.audio)
    else:
        parser.print_help()
        parser.exit()

    # Point to a path containing the pre-trained models & resolve ~ if used
    dirName = os.path.expanduser(args.model)

    # Resolve all the paths of model files
    lm = None
    trie = None
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

    if args.audio is not None:
        title_names = ['Filename', 'Duration(s)', 'Inference Time(s)', 'Model Load Time(s)', 'LM Load Time(s)']
        print("\n%-30s %-20s %-20s %-20s %s" % (title_names[0], title_names[1], title_names[2], title_names[3], title_names[4]))

        inference_time = 0.0

        # Run VAD on the input file
        waveFile = args.audio
        segments, sample_rate, audio_length = wavTranscriber.vad_segment_generator(waveFile, args.aggressive,
                                                                                   model_sample_rate=model_retval[3],
                                                                                   normalize=args.normalize)
        f = open(waveFile.rstrip(".wav") + ".txt", 'w')
        logging.debug("Saving Transcript @: %s" % waveFile.rstrip(".wav") + ".txt")

        if args.save_segments:
            if os.path.isdir("chunks"):
                shutil.rmtree("chunks")
            os.mkdir("chunks")
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
            inference_time += output[1]
            logging.debug("Transcript: %s" % output[0])

            f.write(output[0] + " ")

        if args.save_without_pause:
            whole_audio = segments[0]
            for segment in segments[1:]:
                whole_audio.append(segment)
            new_audio_file = waveFile.rstrip(".wav") + "_no_pause.wav"
            with wave.open(new_audio_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(model_retval[3])
                wf.writeframes(whole_audio)

        # Summary of the files processed
        f.close()

        # Extract filename from the full file path
        filename, ext = os.path.split(os.path.basename(waveFile))
        logging.debug("************************************************************************************************************")
        logging.debug("%-30s %-20s %-20s %-20s %s" % (title_names[0], title_names[1], title_names[2], title_names[3], title_names[4]))
        logging.debug("%-30s %-20.3f %-20.3f %-20.3f %-0.3f" % (filename + ext, audio_length, inference_time, model_retval[1], model_retval[2]))
        logging.debug("************************************************************************************************************")
        print("%-30s %-20.3f %-20.3f %-20.3f %-0.3f" % (filename + ext, audio_length, inference_time, model_retval[1], model_retval[2]))
    else:
        sctx = model_retval[0].createStream()
        if os.name == 'nt': # Windows
            command = 'sox -d -q -V0 -e signed -L -c 1 -b 16 -r 16k -t raw - gain -2'
        else:
            command = 'rec -q -V0 -e signed -L -c 1 -b 16 -r 16k -t raw - gain -2'
        subproc = subprocess.Popen(shlex.split(command),
                                   stdout=subprocess.PIPE,
                                   bufsize=0)
        print('You can start speaking now. Press Control-C to stop recording.')

        try:
            while True:
                data = subproc.stdout.read(512)
                model_retval[0].feedAudioContent(sctx, np.frombuffer(data, np.int16))
        except KeyboardInterrupt:
            print('Transcription: ', model_retval[0].finishStream(sctx))
            subproc.terminate()
            subproc.wait()


if __name__ == '__main__':
    main(sys.argv[1:])
