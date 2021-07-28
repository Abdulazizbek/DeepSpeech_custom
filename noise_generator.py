import warnings

import numpy as np
import scipy.io.wavfile
from numba import jit
from scipy.signal import lfilter

from numpy.fft import irfft

warnings.filterwarnings("ignore")


def wav_read(filename):
    fs, x = scipy.io.wavfile.read(filename)
    if np.issubdtype(x.dtype, np.integer):
        x = x / np.iinfo(x.dtype).max
    return x, fs


def wav_write(filename, s, fs):
    if s.dtype != np.int16:
        s = np.array(s * 2 ** 15, dtype=np.int16)
    if np.any(s > np.iinfo(np.int16).max) or np.any(s < np.iinfo(np.int16).min):
        warnings.warn('Warning: clipping detected when writing {}'.format(filename))
    scipy.io.wavfile.write(filename, fs, s)


@jit
def asl_meter(x, fs, nbits=16):
    """
    Measure the Active Speech Level (ASR) of x following ITU-T P.56.
    If x is integer, it will be scaled to (-1, 1) according to nbits.
    """

    if np.issubdtype(x.dtype, np.integer):
        x = x / 2 ** (nbits - 1)

    # Constants
    T = 0.03  # Time constant of smoothing in seconds
    g = np.exp(-1 / (T * fs))
    H = 0.20  # Time of handover in seconds
    I = int(np.ceil(H * fs))
    M = 15.9  # Margin between threshold and ASL in dB

    a = np.zeros(nbits - 1)  # Activity count
    c = 0.5 ** np.arange(nbits - 1, 0, step=-1)  # Threshold level
    h = np.ones(nbits) * I  # Hangover count
    s = 0
    sq = 0
    p = 0
    q = 0
    asl = -100

    L = len(x)
    s = sum(abs(x))
    sq = sum(x ** 2)
    c_dB = 20 * np.log10(c)

    for i in range(L):
        p = g * p + (1 - g) * abs(x[i])
        q = g * q + (1 - g) * p

        for j in range(nbits - 1):
            if q >= c[j]:
                a[j] += 1
                h[j] = 0
            elif h[j] < I:
                a[j] += 1
                h[j] += 1

    a_dB = -100 * np.ones(nbits - 1)

    for i in range(nbits - 1):
        if a[i] != 0:
            a_dB[i] = 10 * np.log10(sq / a[i])

    delta = a_dB - c_dB
    idx = np.where(delta <= M)[0]

    if len(idx) != 0:
        idx = idx[0]
        if idx > 1:
            asl = bin_interp(a_dB[idx], a_dB[idx - 1], c_dB[idx], c_dB[idx - 1], M)
        else:
            asl = a_dB[idx]

    return asl


@jit
def bin_interp(upcount, lwcount, upthr, lwthr, margin, tol=0.1):
    n_iter = 1
    if abs(upcount - upthr - margin) < tol:
        midcount = upcount
    elif abs(lwcount - lwthr - margin) < tol:
        midcount = lwcount
    else:
        midcount = (upcount + lwcount) / 2
        midthr = (upthr + lwthr) / 2
        diff = midcount - midthr - margin
        while abs(diff) > tol:
            n_iter += 1
            if n_iter > 20:
                tol *= 1.1
            if diff > tol:
                midcount = (upcount + midcount) / 2
                midthr = (upthr + midthr) / 2
            elif diff < -tol:
                midcount = (lwcount + midcount) / 2
                midthr = (lwthr + midthr) / 2
            diff = midcount - midthr - margin
    return midcount


def rms_energy(x):
    return 10 * np.log10((1e-12 + x.dot(x)) / len(x))


def add_noise(speech, noise, fs, snr, asl_level=-26.0):
    """
    Adds noise to a speech signal at a given SNR.
    The speech level is computed as the P.56 active speech level, and
    the noise level is computed as the RMS level. The speech level is considered
    as the reference.
    """
    # Ensure masker is at least as long as signal
    if len(noise) < len(speech):
        raise ValueError('len(noise) needs to be at least equal to len(speech)!')

    # Generate random section of masker
    if len(noise) != len(speech):
        idx = np.random.randint(0, len(noise) - len(speech))
        noise = noise[idx:idx + len(speech)]

    # Scale noise wrt speech at target SNR
    noise_db = rms_energy(noise)

    signal_db = asl_meter(speech, fs)

    # Rescale noise
    noise_new_db = signal_db - snr
    noise_scaled = 10 ** (noise_new_db / 20) * noise / 10 ** (noise_db / 20)

    y = speech + noise_scaled

    y = y / 10 ** (asl_meter(y, fs) / 20) * 10 ** (asl_level / 20)

    return y, noise_scaled, signal_db


def add_reverberation(speech, reverberation, fs, asl_level=-26.0):
    """
    Adds re-verb (convolutive noise) to a speech signal.
    The output speech level is normalized to asl_level.
    """
    y = lfilter(reverberation, 1, speech)
    y = y / 10 ** (asl_meter(y, fs) / 20) * 10 ** (asl_level / 20)

    return y



def ms(x):
    return (np.abs(x) ** 2.0).mean()


def normalize(y, x=None):
    if x is not None:
        x = ms(x)
    else:
        x = 1.0
    return y * np.sqrt(x / ms(y))


def noise(nb_samples, color='white', state=None):
    try:
        return _noise_generators[color](nb_samples, state)
    except KeyError:
        raise ValueError("Incorrect color.")


def white(nb_samples, state=None):
    state = np.random.RandomState() if state is None else state
    return state.randn(nb_samples)


def pink(nb_samples, state=None):
    state = np.random.RandomState() if state is None else state
    uneven = nb_samples % 2
    X = state.randn(nb_samples // 2 + 1 + uneven) + 1j * state.randn(nb_samples // 2 + 1 + uneven)
    S = np.sqrt(np.arange(len(X)) + 1.)  # +1 to avoid divide by zero
    y = (irfft(X / S)).real
    if uneven:
        y = y[:-1]
    return normalize(y)


def blue(nb_samples, state=None):
    state = np.random.RandomState() if state is None else state
    uneven = nb_samples % 2
    X = state.randn(nb_samples // 2 + 1 + uneven) + 1j * state.randn(nb_samples // 2 + 1 + uneven)
    S = np.sqrt(np.arange(len(X)))  # Filter
    y = (irfft(X * S)).real
    if uneven:
        y = y[:-1]
    return normalize(y)


def brown(nb_samples, state=None):
    state = np.random.RandomState() if state is None else state
    uneven = nb_samples % 2
    X = state.randn(nb_samples // 2 + 1 + uneven) + 1j * state.randn(nb_samples // 2 + 1 + uneven)
    S = (np.arange(len(X)) + 1)  # Filter
    y = (irfft(X / S)).real
    if uneven:
        y = y[:-1]
    return normalize(y)


def violet(nb_samples, state=None):
    state = np.random.RandomState() if state is None else state
    uneven = nb_samples % 2
    X = state.randn(nb_samples // 2 + 1 + uneven) + 1j * state.randn(nb_samples // 2 + 1 + uneven)
    S = (np.arange(len(X)))  # Filter
    y = (irfft(X * S)).real
    if uneven:
        y = y[:-1]
    return normalize(y)


_noise_generators = {'white': white,
                     'pink': pink,
                     'blue': blue,
                     'brown': brown,
                     'violet': violet,
                     }
