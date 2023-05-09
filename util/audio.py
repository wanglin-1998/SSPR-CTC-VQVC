import librosa
import librosa.filters
import numpy as np
from scipy import signal
from scipy.io import wavfile


def load_wav(path, expect_sr):
    audio, sr = librosa.core.load(path, sr=None)
    if sr != expect_sr:
        raise ValueError("sampling rate mismatch: expected {0} but met with {1}".format(expect_sr, sr))
    return audio
    #return librosa.core.load(path, sr=expect_sr)[0]

def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    #proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))

def save_wavenet_wav(wav, path, sr):
    librosa.output.write_wav(path, wav, sr=sr)

#From https://github.com/r9y9/wavenet_vocoder/blob/master/audio.py
def start_and_end_indices(quantized, silence_threshold=2):
    for start in range(quantized.size):
        if abs(quantized[start] - 127) > silence_threshold:
            break
    for end in range(quantized.size - 1, 1, -1):
        if abs(quantized[end] - 127) > silence_threshold:
            break

    assert abs(quantized[start] - 127) > silence_threshold
    assert abs(quantized[end] - 127) > silence_threshold

    return start, end

def trim_silence(wav, condition):
    '''Trim leading and trailing silence

    Useful for M-AILABS dataset if we choose to trim the extra 0.5 silence at beginning and end.
    '''
    #Thanks @begeekmyfriend and @lautjy for pointing out the params contradiction. These params are separate and tunable per dataset.
    return librosa.effects.trim(wav, top_db=condition['trim_top_db'], frame_length=condition['trim_fft_size'],
                                hop_length=condition['trim_hop_size'])[0]

def get_hop_size(condition):
    hop_size = condition['hop_size']
    #if hop_size is None:
    #    assert condition['frame_shift_ms'] is not None
    #    hop_size = int(condition['sample_rate'] / 1000 * condition['frame_shift_ms'])
    return hop_size

def linearspectrogram(wav, condition):
    D = _stft(wav, condition)
    S = _amp_to_db(np.abs(D), condition) - condition['ref_level_db']

    if condition['signal_normalization']:
        return _normalize(S, condition)
    return S

def melspectrogram(wav, condition):
    D = _stft(wav, condition)
    S = _amp_to_db(_linear_to_mel(np.abs(D), condition), condition) - condition['ref_level_db']

    if condition['signal_normalization']:
        return _normalize(S, condition)
    return S

def inv_linear_spectrogram(linear_spectrogram, condition):
    '''Converts linear spectrogram to waveform using librosa'''
    if condition['signal_normalization']:
        D = _denormalize(linear_spectrogram, condition)
    else:
        D = linear_spectrogram

    S = _db_to_amp(D + condition['ref_level_db']) #Convert back to linear

    if condition['use_lws']:
        processor = _lws_processor(condition)
        D = processor.run_lws(S.astype(np.float64).T ** condition['power'])
        y = processor.istft(D).astype(np.float32)
        return y
    else:
        return _griffin_lim(S ** condition['power'], condition)

def inv_mel_spectrogram(mel_spectrogram, condition):
    '''Converts mel spectrogram to waveform using librosa'''
    if condition['signal_normalization']:
        D = _denormalize(mel_spectrogram, condition)
    else:
        D = mel_spectrogram

    S = _mel_to_linear(_db_to_amp(D + condition['ref_level_db']), condition)  # Convert back to linear

    if condition['use_lws']:
        processor = _lws_processor(condition)
        D = processor.run_lws(S.astype(np.float64).T ** condition['power'])
        y = processor.istft(D).astype(np.float32)
        return y
    else:
        return _griffin_lim(S ** condition['power'], condition)

def _lws_processor(condition):
    import lws
    return lws.lws(condition['n_fft'], get_hop_size(condition), fftsize=condition['win_size'], mode="speech")

def _griffin_lim(S, condition):
    '''librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    '''
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles, condition)
    for i in range(condition['griffin_lim_iters']):
        angles = np.exp(1j * np.angle(_stft(y, condition)))
        y = _istft(S_complex * angles, condition)
    return y

def _stft(y, condition):
    if condition['use_lws']:
        return _lws_processor(condition).stft(y).T
    else:
        return librosa.stft(y=y, n_fft=condition['n_fft'], hop_length=get_hop_size(condition), win_length=condition['win_size'])

def _istft(y, condition):
    return librosa.istft(y, hop_length=get_hop_size(condition), win_length=condition['win_size'])

def num_frames(length, fsize, fshift):
    """Compute number of time frames of spectrogram
    """
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M

def pad_lr(x, fsize, fshift):
    """Compute left and right padding
    """
    M = num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r


# Conversions
_mel_basis = None
_inv_mel_basis = None

def _linear_to_mel(spectogram, condition):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(condition)
    return np.dot(_mel_basis, spectogram)

def _mel_to_linear(mel_spectrogram, condition):
    global _inv_mel_basis
    if _inv_mel_basis is None:
        _inv_mel_basis = np.linalg.pinv(_build_mel_basis(condition))
    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))

def _build_mel_basis(condition):
    #assert condition['fmax'] <= condition['sample_rate'] // 2
    return librosa.filters.mel(condition['sample_rate'], condition['n_fft'], n_mels=condition['num_mels'],
                               fmin=30, fmax=condition['sample_rate']//2-100)
                               #fmin=condition.fmin, fmax=condition.fmax)

def _amp_to_db(x, condition):
    min_level = np.exp(condition['min_level_db'] / 20 * np.log(10)) # 1e-5 is -100db
    return 20 * np.log10(np.maximum(min_level, x))

def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)

def _normalize(S, condition):
    if condition['allow_clipping_in_normalization']:
        if condition['symmetric_mels']:
            return np.clip((2 * condition['max_abs_value']) * ((S - condition['min_level_db']) / (-condition['min_level_db'])) - condition['max_abs_value'],
             -condition['max_abs_value'], condition['max_abs_value'])
        else:
            return np.clip(condition['max_abs_value'] * ((S - condition['min_level_db']) / (-condition['min_level_db'])), 0, condition['max_abs_value'])

    # Belowing code has problem with 'condition.min_level_db'
    #assert S.max() <= 0 and S.min() - condition.min_level_db >= 0
    #if condition.symmetric_mels:
    #    return (2 * condition.max_abs_value) * ((S - condition.min_level_db) / (-condition.min_level_db)) - condition.max_abs_value
    #else:
    #    return condition.max_abs_value * ((S - condition.min_level_db) / (-condition.min_level_db))

    # <wwf>
    min_level_db = condition['min_level_db'] - condition['ref_level_db']
    #assert S.max() <= 0
    assert S.min() >= min_level_db or abs(S.min() - min_level_db) <= 1e-8
    if condition['symmetric_mels']:
        # not strict [-condition.max_abs_value, condition.max_abs_value]
        return (2 * condition['max_abs_value']) * ((S - min_level_db) / (-min_level_db)) - condition['max_abs_value']
    else:
        # not strict [0, condition.max_abs_value], upper bound is possibly greater than condition.max_abs_value
        return condition['max_abs_value'] * ((S - min_level_db) / (-min_level_db))

def _denormalize(D, condition):
    if condition['allow_clipping_in_normalization']:
        if condition['symmetric_mels']:
            return (((np.clip(D, -condition['max_abs_value'],
                condition['max_abs_value']) + condition['max_abs_value']) * -condition['min_level_db'] / (2 * condition['max_abs_value']))
                + condition['min_level_db'])
        else:
            return ((np.clip(D, 0, condition['max_abs_value']) * -condition['min_level_db'] / condition['max_abs_value']) + condition['min_level_db'])

    # Belowing code has problem with 'condition.min_level_db'
    #if condition.symmetric_mels:
    #    return (((D + condition.max_abs_value) * -condition.min_level_db / (2 * condition.max_abs_value)) + condition.min_level_db)
    #else:
    #    return ((D * -condition.min_level_db / condition.max_abs_value) + condition.min_level_db)

    # <wwf>
    min_level_db = condition['min_level_db'] - condition['ref_level_db']
    if condition['symmetric_mels']:
        return (((D + condition['max_abs_value']) * -min_level_db / (2 * condition['max_abs_value'])) + min_level_db)
    else:
        return ((D * -min_level_db / condition['max_abs_value']) + min_level_db)
