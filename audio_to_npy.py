import os
import sys
import pydub
import numpy as np
from pydub import effects
import python_speech_features


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
    return complex_spec


def process_data(file_list, output_path):
    ret = []
    for filename in file_list:
        try:
            signal = load_file(filename, 'wav')
            complex_spec = fft_singal(signal, None)
            ret.append(complex_spec)
        except:
            pass

    ret = np.array(ret)
    np.save(output_path, ret)


def walk_path(base_path):
    ret = []
    for root, directory, files in os.walk(base_path):
        ret.extend([os.path.join(root, filename) for filename in files])
    return ret


if __name__ == '__main__':
    file_list = walk_path(sys.argv[1])
    output_file = sys.argv[2]
    process_data(file_list, output_file)
