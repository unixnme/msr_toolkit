from subprocess import call
import glob
import sys
from os.path import join

# convert to wave files
filenames = glob.glob('**/*.flac', recursive=True)
to_write = ''
for f in filenames:
    target = f.replace('.flac', '.wav')
    #call(['ffmpeg', '-i', f, target, '-y'])
    #call(['ffmpeg', '-i', f, '-f', 's16le', '-acodec', 'pcm_s16le', target, '-y'])
    to_write += target + '\t' + target.replace('.wav', '.mfc') + '\n'

with open('src.lst', 'w') as f:
    f.write(to_write)

# convert to mfc files
call(['HCopy', '-A', '-C', 'mfc.conf', '-S', 'src.lst'])

