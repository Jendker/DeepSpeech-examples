# based on evaluate_tflite.py

import csv
import sys
import os
import logging
import argparse
import time

import numpy as np
import wavTranscriber
from deepspeech import Model
from multiprocessing import JoinableQueue, Process, cpu_count, Manager, Pool


BEAM_WIDTH = 500
LM_ALPHA = 0.75
LM_BETA = 1.85


# Debug helpers
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def worker_pool(model, lm, trie, gpu_mask, aggressive, normalize, filenames):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_mask)
    ds = Model(model, BEAM_WIDTH)
    ds.enableDecoderWithLM(lm, trie, LM_ALPHA, LM_BETA)
    all_files_output = []

    for filename in filenames:
        try:
            wavname = os.path.splitext(os.path.basename(filename))[0]
            segments, sample_rate, audio_length = wavTranscriber.vad_segment_generator(filename, aggressive,
                                                                                       model_sample_rate=ds.sampleRate(),
                                                                                       normalize=normalize)
            output_full = ''
            for i, segment in enumerate(segments):
                # Run deepspeech on the chunk that just completed VAD
                logging.debug("Processing chunk %002d" % (i,))
                audio = np.frombuffer(segment, dtype=np.int16)
                output = ds.stt(audio)

                output_full += output + " "

            all_files_output.append({'wav': wavname, 'prediction': output_full})
        except FileNotFoundError as ex:
            print('FileNotFoundError: ', ex)


def worker(model, lm, trie, queue_in, queue_out, gpu_mask, aggressive, normalize):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_mask)
    ds = Model(model, BEAM_WIDTH)
    ds.enableDecoderWithLM(lm, trie, LM_ALPHA, LM_BETA)

    while True:
        try:
            msg = queue_in.get()

            filename = msg['filename']
            wavname = os.path.splitext(os.path.basename(filename))[0]
            segments, sample_rate, audio_length = wavTranscriber.vad_segment_generator(filename, aggressive,
                                                                                       model_sample_rate=ds.sampleRate(),
                                                                                       normalize=normalize)
            output_full = ''
            for i, segment in enumerate(segments):
                # Run deepspeech on the chunk that just completed VAD
                logging.debug("Processing chunk %002d" % (i,))
                audio = np.frombuffer(segment, dtype=np.int16)
                output = ds.stt(audio)

                output_full += output + " "

            queue_out.put({'wav': wavname, 'prediction': output_full})
        except FileNotFoundError as ex:
            print('FileNotFoundError: ', ex)

        print(queue_out.qsize(), end='\r')  # Update the current progress
        queue_in.task_done()


def main():
    parser = argparse.ArgumentParser(description='Transcribe long audio files using webRTC VAD or use the streaming interface')
    parser.add_argument('--aggressive', type=int, choices=range(4), required=False,
                        help='Determines how aggressive filtering out non-speech is. (Interger between 0-3)')
    parser.add_argument('--input_path', required=True,
                        help='Path to csv file with list of files')
    parser.add_argument('--model', required=True,
                        help='Path to the model (protocol buffer binary file)')
    parser.add_argument('--lm', required=True,
                        help='Path to the language model binary file')
    parser.add_argument('--trie', required=True,
                        help='Path to the language model trie file created with native_client/generate_trie')
    parser.add_argument('--proc', required=False, default=cpu_count(), type=int,
                        help='Number of processes to spawn, defaulting to number of CPUs')
    parser.add_argument('--normalize', dest='normalize', action='store_true')
    parser.add_argument('--no_normalize', dest='normalize', action='store_false')
    parser.add_argument('--pool', action='store_true')
    parser.set_defaults(normalize=True)
    args = parser.parse_args()

    start_time = time.time()

    if args.pool:
        print("Running pool of subprocesses.")
        with Pool(args.proc) as pool:
            all_files = []
            with open(args.input_path, 'r') as csvfile:
                csvreader = csv.DictReader(csvfile)
                count = 0
                for row in csvreader:
                    count += 1
                    all_files.append(row['filename'])

            split_filenames = chunkIt(all_files, args.proc)
            print('Totally %d wav entries found in csv\n' % count)
            workers_arguments = []
            for index, process_filenames in enumerate(split_filenames):
                    workers_arguments.append((args.model, args.lm, args.trie, index, args.aggressive, args.normalize, process_filenames))
            transcripts = pool.starmap(worker_pool, workers_arguments)
            print('\nTotally %d wav file transcripted' % len(transcripts))
    else:
        print("Spawning processes.")
        manager = Manager()
        work_todo = JoinableQueue()  # this is where we are going to store input data
        work_done = manager.Queue()  # this where we are gonna push them out

        processes = []
        for i in range(args.proc):
            worker_process = Process(target=worker, args=(args.model, args.lm, args.trie, work_todo, work_done, i,
                                                          args.aggressive, args.normalize),
                                     daemon=True, name='worker_process_{}'.format(i))
            worker_process.start()  # Launch reader() as a separate python process
            processes.append(worker_process)

        print([x.name for x in processes])

        predictions = []
        wav_filenames = []

        with open(args.input_path, 'r') as csvfile:
            csvreader = csv.DictReader(csvfile)
            count = 0
            for row in csvreader:
                count += 1
                work_todo.put({'filename': row['filename']})
                wav_filenames.extend(row['filename'])

        print('Totally %d wav entries found in csv\n' % count)
        work_todo.join()
        print('\nTotally %d wav file transcripted' % work_done.qsize())

        while not work_done.empty():
            msg = work_done.get()
            predictions.append(msg['prediction'])

    print("Runtime:", time.time() - start_time)


if __name__ == '__main__':
    main()
