import os
import sys

import h5py
import pydub
import numpy as np
from pydub import effects
import python_speech_features
from multiprocessing import Pool


def load_file(filename, file_format, frame_rate=16000):
    sound = pydub.AudioSegment.from_file(filename, file_format)
    sound = sound.set_frame_rate(frame_rate)
    sound = sound.set_channels(1)
    sound = sound.set_sample_width(2)
    sound = sound.remove_dc_offset()
    sound = effects.normalize(sound)
    return np.array(sound.get_array_of_samples())


def fft_singal(singal, pre_frame, window_size=512, shift_size=160, window_func=(lambda x: np.ones((x,))), nfft=512):
    singal = pre_frame(singal) if pre_frame is not None else singal
    frames = python_speech_features.sigproc.framesig(singal, window_size, shift_size, window_func)
    complex_spec = np.fft.rfft(frames, nfft)
    return complex_spec.astype('complex64')


def fbank_from_complex_spec(complex_spec, nfilt=64, nfft=512, sample_rate=16000):
    power = 1 / nfft * np.square(complex_spec).real
    fb = python_speech_features.get_filterbanks(nfilt, nfft, sample_rate)
    feat = np.dot(power, fb.T)
    feat = np.where(feat == 0, np.finfo(float).eps, feat)
    return feat.astype('float32')


def dleta_fbank(feat):
    last = np.zeros(feat[0].shape)
    ret = np.zeros(feat.shape)
    for item, idx in zip(feat, range(feat.shape[0])):
        dleta = item - last
        ret[idx, :] = dleta
    return ret.astype('float32')


def process_data_single(filename):
        try:
            signal = load_file(filename, 'wav')
            complex_spec = fft_singal(signal, None)
            fbank = fbank_from_complex_spec(complex_spec, 64, 512)
            dleta1 = dleta_fbank(fbank)
            dleta2 = dleta_fbank(dleta1)
            return [filename, complex_spec, fbank, dleta1, dleta2]
        except Exception as e:
            print('[error]', filename, e)
            return None


def process_data(file_list, output_path):
    output_file = h5py.File(output_path, 'w')

    def save_data(args):
        if args is None:
            return
        [filename, complex_spec, fbank, dleta1, dleta2] = args
        output_file[filename + '_spec'] = complex_spec
        output_file[filename + '_fbank'] = fbank
        output_file[filename + '_dleta1'] = dleta1
        output_file[filename + '_dleta2'] = dleta2

    with Pool(processes=3) as pool:
        pool.map_async(process_data_single, file_list, callback=save_data)


def walk_path(base_path):
    ret = []
    for root, directory, files in os.walk(base_path):
        ret.extend([os.path.join(root, filename) for filename in files])
    return ret


if __name__ == '__main__':
    file_list = walk_path(sys.argv[1])
    output_path = sys.argv[2]
    process_data(file_list, output_path)
