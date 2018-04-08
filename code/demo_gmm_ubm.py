import hmmlearn as hmm
import numpy as np
from struct import unpack

def htkread(filename, endian='little'):
    if endian == 'big':
        op = '>'
    elif endian == 'little':
        op = '<'
    else:
        raise ValueError('endian must be either big or little')

    with open(filename, 'rb') as f:
        data = f.read(4)
        data = unpack(op + 'i', data)
        nframes = data[0]
        if nframes <= 0:
            raise ValueError('nframes is nonpositive; check file or endian')

        data = f.read(4)
        data = unpack(op + 'i', data)
        frate = data[0]
        if frate <= 0:
            raise ValueError('frate is nonpositive; check file or endian')

        data = f.read(2)
        data = unpack(op + 'h', data)
        nbytes = data[0]
        if nbytes <= 0:
            raise ValueError('nbytes is nonpositive; check file or endian')

        data = f.read(2)
        data = unpack(op + 'h', data)
        feakind = data[0]

        ndim = nbytes // 4
        data = np.fromfile(f, dtype=op + 'f4').reshape(ndim, -1)

    return data, frate, feakind


def gmm_em(dataList, nmix, final_niter, ds_factor):
    pass

dataList = '../ubm.lst'
nmix = 256
final_niter = 10
ds_factor = 1
ubm = gmm_em(dataList, nmix, final_niter, ds_factor)

data, frate, feakind = htkread('../LibriSpeech/dev-clean/2277/149874/2277-149874-0002.mfc')

