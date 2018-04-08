from sklearn.mixture import GaussianMixture
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
        # this part is so crucial!!
        data = np.fromfile(f, dtype=op + 'f4').reshape(-1, ndim).T

    return data, frate, feakind


def load_data(dataList):
    if isinstance(dataList, str):
        with open(dataList, 'r') as f:
            filenames = f.read().split()
    else:
        filenames = dataList

    data = []
    for i,filename in enumerate(filenames):
        d, _, _ = htkread(filename)
        data.append(d)

    return data

def comp_gm_gv(dataList):
    gm = np.zeros(dataList[0].shape[0])
    gv = np.zeros(dataList[0].shape[0])
    nframes = 0
    data = np.concatenate(dataList, axis=1)
    gm = np.mean(data, axis=1)
    gv = np.var(data, axis=1)

    return gm, gv

def gmm_em(dataList, nmix, final_niter, ds_factor):
    dataList = load_data(dataList)
    nfiles = len(dataList)
    gm, gv = comp_gm_gv(dataList)

    model = GaussianMixture(1, 'diag')
    data = np.concatenate(dataList, axis=1).T
    niter = [1, 2, 4, 4, 4, 4, 6, 6, 10, 10, 15]
    niter[int(np.log2(nmix))] = final_niter

    mix = 1
    while mix <= nmix:
        if mix >= nmix//2:
            ds_factor = 1

        print('Re-estimating the GMM hyperparameters for %d components ...' % mix)
        for i in range(niter[int(np.log2(mix))]):
            print('EM iter#: %d \t' % i, end='')
            model.fit(data)
            llk,_ = model._estimate_log_prob_resp(data)
            print('[llk = %.2f]' % np.mean(llk))

        if mix < nmix:
            mix *= 2
            gm = model.means_
            gm = np.concatenate([gm, gm])
            model = GaussianMixture(mix, 'diag', means_init=gm)
            
    pass

dataList = '../ubm.lst'
nmix = 256
final_niter = 10
ds_factor = 1
ubm = gmm_em(dataList, nmix, final_niter, ds_factor)

data = load_data(dataList)

