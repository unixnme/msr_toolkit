import os
import glob
import numpy as np
np.random.seed(0)

# find all mfc files
DIR = '../LibriSpeech'
files = glob.glob(DIR + '/**/*.mfc', recursive=True)

# collect by user
speakers = {}
for filepath in files:
    spkr = filepath.split('/')[-3]
    if spkr not in speakers:
        speakers[spkr] = []
    speakers[spkr].append(filepath)

# choose N users for test / train
N = len(speakers) // 10
speaker_list = np.asarray(sorted(list(speakers.keys())))
np.random.shuffle(speaker_list)

# select M files for train
train_files = {}
test_files = {}
M = 5
for spkr in speaker_list[:N]:
    # these are train/test speakers
    files = speakers[spkr]
    np.random.shuffle(files)
    train_files[spkr] = files[:M]
    test_files[spkr] = files[M:]

ubm_files = {}
for spkr in speaker_list[N:]:
    # these are UBM speakers
    files = speakers[spkr]
    ubm_files[spkr] = files

ubm_to_write = ''
ubm_ind_to_write = ''
for spkr in speaker_list[N:]:
    files = ubm_files[spkr]
    for filepath in files:
        ubm_to_write += filepath + '\n'
        ubm_ind_to_write += spkr + ' ' + filepath + '\n'

with open('ubm.lst', 'w') as f:
    f.write(ubm_to_write)

with open('ubm_ind.lst', 'w') as f:
    f.write(ubm_ind_to_write)

with open('train.lst', 'w') as f:
    for spkr in speaker_list[:N]:
        for filepath in train_files[spkr]:
            f.write(spkr + ' ' + filepath + '\n')

with open('test.lst', 'w') as f:
    for spkr in speaker_list[:N]:
        for trial in speaker_list[:N]:
            files = test_files[spkr]
            if spkr == trial: label = 'target'
            else: label = 'imposter'
            for filepath in files:
                f.write(spkr + ' ' + filepath + ' ' + label + '\n')

