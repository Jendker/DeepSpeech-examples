# based on evaluate_tflite.py

import csv
import sys
import os
import logging
import argparse
import time
import pickle

import numpy as np
import wavTranscriber
from deepspeech import Model
from multiprocessing import JoinableQueue, Process, cpu_count, Manager, Pool


BEAM_WIDTH = 500
LM_ALPHA = 0.75
LM_BETA = 1.85


# Debug helpers
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)


def process_audio_file(audio_path, ds, aggressive, normalize):
    segments, sample_rate, audio_length = wavTranscriber.vad_segment_generator(audio_path, aggressive,
                                                                               model_sample_rate=ds.sampleRate(),
                                                                               normalize=normalize)
    full_text = []
    for i, segment in enumerate(segments):
        # Run deepspeech on the chunk that just completed VAD
        logging.debug("Processing chunk %002d" % (i,))
        audio = np.frombuffer(segment, dtype=np.int16)
        output = ds.stt(audio)
        logging.debug("Transcript: %s" % output)
        full_text += output + " "
    return full_text


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def pool_worker(model, lm, trie, gpu_mask, aggressive, normalize, file_paths, use_lm, use_pb):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_mask)
    if use_pb:
        if '.pbmm' in model:
            model = model[:-2]
    else:
        if '.pb' == model[:-3]:
            model += 'mm'

    ds = Model(model, BEAM_WIDTH)
    if use_lm:
        ds.enableDecoderWithLM(lm, trie, LM_ALPHA, LM_BETA)
    all_files_output = []

    for file_path in file_paths:
        try:
            prediction = process_audio_file(file_path, ds, aggressive, normalize)
            wavname = os.path.splitext(os.path.basename(file_path))[0]
            all_files_output.append({'wav': wavname, 'prediction': prediction})
        except FileNotFoundError as ex:
            print('FileNotFoundError: ', ex)


def proc_worker(model, lm, trie, queue_in, queue_out, gpu_mask, aggressive, normalize, use_lm, use_pb):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_mask)
    if use_pb:
        if '.pbmm' in model:
            model = model[:-2]
    else:
        if '.pb' == model[:-3]:
            model += 'mm'
    ds = Model(model, BEAM_WIDTH)
    if use_lm:
        ds.enableDecoderWithLM(lm, trie, LM_ALPHA, LM_BETA)

    while True:
        try:
            msg = queue_in.get()

            file_path = msg['filename']
            prediction = process_audio_file(file_path, ds, aggressive, normalize)
            wavname = os.path.splitext(os.path.basename(file_path))[0]
            queue_out.put({'wav': wavname, 'prediction': prediction})
        except FileNotFoundError as ex:
            print('FileNotFoundError: ', ex)

        print(queue_out.qsize(), end='\r')  # Update the current progress
        queue_in.task_done()


def run(pool, use_lm, proc, input_path, model, lm, trie, aggressive, normalize, use_pb):
    start_time = time.time()
    if pool:
        print("Running pool of subprocesses.")
        with Pool(proc) as pool:
            all_files = []
            with open(input_path, 'r') as csvfile:
                csvreader = csv.DictReader(csvfile)
                count = 0
                for row in csvreader:
                    count += 1
                    all_files.append(row['filename'])

            split_filenames = chunkIt(all_files, proc)
            print('Totally %d wav entries found in csv\n' % count)
            workers_arguments = []
            for index, process_filenames in enumerate(split_filenames):
                    workers_arguments.append((model, lm, trie, index, aggressive, normalize, process_filenames, use_lm,
                                              use_pb))
            transcripts = pool.starmap(pool_worker, workers_arguments)
            print('\nTotally %d wav file transcripted' % len(transcripts))
    else:
        print("Spawning processes.")
        manager = Manager()
        work_todo = JoinableQueue()  # this is where we are going to store input data
        work_done = manager.Queue()  # this where we are gonna push them out

        processes = []
        for i in range(proc):
            worker_process = Process(target=proc_worker, args=(model, lm, trie, work_todo, work_done, i,
                                                               aggressive, normalize, use_lm, use_pb),
                                     daemon=True, name='worker_process_{}'.format(i))
            worker_process.start()  # Launch reader() as a separate python process
            processes.append(worker_process)

        print([x.name for x in processes])

        predictions = []
        wav_filenames = []

        with open(input_path, 'r') as csvfile:
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

        for process in processes:
            process.terminate()

    runtime = time.time() - start_time
    print("Runtime:", runtime)
    return runtime


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
    parser.add_argument('--use_lm', dest='use_lm', action='store_true')
    parser.add_argument('--no_use_lm', dest='use_lm', action='store_false')
    parser.add_argument('--use_pb', dest='use_pb', action='store_true')
    parser.add_argument('--no_use_pb', dest='use_pb', action='store_false')
    parser.add_argument('--pool', action='store_true',
                        help='Use subprocess pool instead of process spawning. There is no big difference between both.')
    parser.add_argument('--benchmark', action='store_true',
                        help='Perform the benchmark')
    parser.set_defaults(normalize=True)
    parser.set_defaults(use_lm=True)
    parser.set_defaults(use_pb=True)
    args = parser.parse_args()

    if args.benchmark:
        print("Running benchmark")
        results = {}
        use_lm_array = [True]
        proc_array = [1, 5, 10, 15, 20]
        use_pb_array = [True, False]
        runs = 3
        all_runs_count = len(use_lm_array) * len(proc_array) * len(use_pb_array) * runs
        count = 0
        for run_no in range(runs):
            for use_lm in use_lm_array:
                for proc in proc_array:
                    for use_pb in use_pb_array:
                        print("Run", count, "/", all_runs_count)
                        count += 1
                        key = (use_lm, proc, use_pb)
                        if key not in results:
                            results[key] = 0
                        this_run_time = run(pool=args.pool, use_lm=use_lm, proc=proc,
                                           input_path=args.input_path, model=args.model,
                            lm=args.lm, trie=args.trie, aggressive=args.aggressive, normalize=args.normalize,
                            use_pb=use_pb)
                        results[key] = results[key] + (this_run_time - results[key]) / (run_no + 1)

        print("Job finished")
        with open('results.pickle', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # with open('filename.pickle', 'rb') as handle:
        #     b = pickle.load(handle)

    else:
        run(args.pool, args.use_lm, args.proc, args.input_path, args.model, args.lm,
            args.trie, args.aggressive, args.normalize, args.use_pb)

if __name__ == '__main__':
    main()
