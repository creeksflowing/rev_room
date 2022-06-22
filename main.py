import numpy as np
import acoustics
import time
import scipy.io
import glob
import os
import acoustics.octave
import acoustics.bands
from tkinter import *
from tkinter import filedialog
# import soundfile as sf
import matplotlib.pyplot as plt
from scipy.sparse import spdiags
from scipy.signal import butter, lfilter, freqz, filtfilt, sosfilt
from scipy.signal import hilbert
from scipy.io import wavfile
from scipy import signal
from numba import jit
from scipy import stats
from acoustics.utils import _is_1d
from acoustics.signal import bandpass
from acoustics.bands import (_check_band_type, octave_low, octave_high, third_low, third_high)
from pathlib import Path
from acoustics.standards.iso_tr_25417_2007 import REFERENCE_PRESSURE
from acoustics.standards.iec_61672_1_2013 import (NOMINAL_OCTAVE_CENTER_FREQUENCIES,
                                                  NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES)

try:
    from pyfftw.interfaces.numpy_fft import rfft
except ImportError:
    from numpy.fft import rfft

OCTAVE_CENTER_FREQUENCIES = NOMINAL_OCTAVE_CENTER_FREQUENCIES
THIRD_OCTAVE_CENTER_FREQUENCIES = NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES

sample_rate = 48000
sweep_duration = 15
starting_frequency = 20.0
ending_frequency = 24000.0
number_of_repetitions = 3


@jit(nopython=True)
def fade(data, gain_start,
         gain_end):
    """
    Create a fade on an input object

    Parameters
    ----------
    data : object
       An input object

    gain_start : scalar
        Fade starting point

    gain_end : scalar
        Fade end point

    Returns
    -------
    data : object
        An input object with the fade applied
    """
    gain = gain_start
    delta = (gain_end - gain_start) / (len(data) - 1)
    for i in range(len(data)):
        data[i] = data[i] * gain
        gain = gain + delta

    return data


@jit(nopython=True)
def generate_exponential_sweep(time_in_seconds, sr):
    """
    Generate an exponential sweep using Farina's log sweep theory

    Parameters
    ----------
    time_in_seconds : scalar
       Duration of the sweep in seconds

    sr : scalar
        The sampling frequency

    Returns
    -------
    exponential_sweep : array
        An array with the fade() function applied
    """
    time_in_samples = time_in_seconds * sr
    exponential_sweep = np.zeros(time_in_samples, dtype=np.double)
    for n in range(time_in_samples):
        t = n / sr
        exponential_sweep[n] = np.sin(
            (2.0 * np.pi * starting_frequency * sweep_duration)
            / np.log(ending_frequency / starting_frequency)
            * (np.exp((t / sweep_duration) * np.log(ending_frequency / starting_frequency)) - 1.0))

    number_of_samples = 50
    exponential_sweep[-number_of_samples:] = fade(exponential_sweep[-number_of_samples:], 1, 0)

    return exponential_sweep


@jit(nopython=True)
def generate_inverse_filter(time_in_seconds, sr,
                            exponential_sweep):
    """
    Generate an inverse filter using Farina's log sweep theory

    Parameters
    ----------
    time_in_seconds : scalar
        Duration of the sweep in seconds

    sr : scalar
        The sampling frequency

    exponential_sweep : array
        The result of the generate_exponential_sweep() function


    Returns
    -------
    inverse_filter : array
         The array resulting from applying an amplitude envelope to the exponential_sweep array
    """
    time_in_samples = time_in_seconds * sr
    amplitude_envelope = np.zeros(time_in_samples, dtype=np.double)
    inverse_filter = np.zeros(time_in_samples, dtype=np.double)
    for n in range(time_in_samples):
        amplitude_envelope[n] = pow(10, (
                (-6 * np.log2(ending_frequency / starting_frequency)) * (n / time_in_samples)) * 0.05)
        inverse_filter[n] = exponential_sweep[-n] * amplitude_envelope[n]

    return inverse_filter


def deconvolve(ir_sweep, ir_inverse):
    """
    A deconvolution of the exponential sweep and the relative inverse filter

    Parameters
    ----------
    ir_sweep : array
        The result of the generate_exponential_sweep() function

    ir_inverse : array
        The result of the generate_inverse_filter() function

    Returns
    -------
    normalized_ir : array
         An N-dimensional array containing a subset of the discrete linear deconvolution of ir_sweep with ir_inverse
    """
    impulse_response = signal.fftconvolve(ir_sweep, ir_inverse,
                                          mode='full')

    normalized_ir = impulse_response * (1.0 / np.max(abs(impulse_response)))

    return normalized_ir


def third(first, last):
    """
    Generate a Numpy array for central frequencies of third octave bands.

    Parameters
    ----------
    first : scalar
       First third octave center frequency.

    last : scalar
        Last third octave center frequency.

    Returns
    -------
    octave_bands : array
        An array of center frequency third octave bands.
    """
    return acoustics.signal.OctaveBand(fstart=first, fstop=last, fraction=3).nominal


def t60_impulse(file, bands,
                rt='t30'):  # pylint: disable=too-many-locals
    """
    Reverberation time from a WAV impulse response.

    Parameters
    ----------
    file: .wav
        Name of the WAV file containing the impulse response.

    bands: array
        Octave or third bands as NumPy array.

    rt: instruction
        Reverberation time estimator. It accepts `'t30'`, `'t20'`, `'t10'` and `'edt'`.

    Returns
    -------
    t60: array
        Reverberation time :math:`T_{60}`
    """
    fs, raw_signal = wavfile.read(file)
    band_type = _check_band_type(bands)

    if band_type == 'octave':
        low = octave_low(bands[0], bands[-1])
        high = octave_high(bands[0], bands[-1])
    elif band_type == 'third':
        low = third_low(bands[0], bands[-1])
        high = third_high(bands[0], bands[-1])

    rt = rt.lower()
    if rt == 't30':
        init = -5.0
        end = -35.0
        factor = 2.0
    elif rt == 't20':
        init = -5.0
        end = -25.0
        factor = 3.0
    elif rt == 't10':
        init = -5.0
        end = -15.0
        factor = 6.0
    elif rt == 'edt':
        init = 0.0
        end = -10.0
        factor = 6.0

    t60 = np.zeros(bands.size)

    for band in range(bands.size):
        # Filtering signal
        filtered_signal = bandpass(raw_signal, low[band], high[band], fs, order=8)
        abs_signal = np.abs(filtered_signal) / np.max(np.abs(filtered_signal))

        # Schroeder integration
        sch = np.cumsum(abs_signal[::-1] ** 2)[::-1]
        sch_db = 10.0 * np.log10(sch / np.max(sch))

        # Linear regression
        sch_init = sch_db[np.abs(sch_db - init).argmin()]
        sch_end = sch_db[np.abs(sch_db - end).argmin()]
        init_sample = np.where(sch_db == sch_init)[0][0]
        end_sample = np.where(sch_db == sch_end)[0][0]
        x = np.arange(init_sample, end_sample + 1) / fs
        y = sch_db[init_sample:end_sample + 1]
        slope, intercept = stats.linregress(x, y)[0:2]

        # Reverberation time (T30, T20, T10 or EDT)
        db_regress_init = (init - intercept) / slope
        db_regress_end = (end - intercept) / slope
        t60[band] = factor * (db_regress_end - db_regress_init)

    return t60


def sabine_absorption_area(sabine_t60, volume):
    """
    Equivalent absorption surface or area in square meters following Sabine's formula

    Parameters
    ----------
    sabine_t60: array
        The result of the t60_impulse() function

    volume: scalar
        The volume of the room

    Returns
    -------
    absorption_area: scalar
        The equivalent absorption surface
    """
    absorption_area = (0.161 * volume) / sabine_t60

    return absorption_area


"""
def file_save(file_name):
    if os.path.isfile(file_name):
        expand = 1
        while True:
            expand += 1
            new_file_name = file_name.split(".txt")[0] + str(expand) + ".txt"
            if os.path.isfile(new_file_name):
                continue
            else:
                file_name = new_file_name
                break


def new_directory(directory, filename):
    # Before creating a new directory, check to see if it already exists
    if not os.path.isdir(directory):
        os.mkdir(directory)

    # Create the new file inside the new directory
    os.chdir(directory)
    with open(filename, "w") as file:
        pass
    os.chdir("..")
    # Return the list of files in the new directory
    return os.listdir(directory)
"""


for n in range(1, number_of_repetitions + 1):
    sweep = generate_exponential_sweep(sweep_duration, sample_rate)
    inverse = generate_inverse_filter(sweep_duration, sample_rate, sweep)
    ir = deconvolve(sweep, inverse)

    sweep_name = "exponential_sweep_%d.wav" % (n,)
    inverse_filter_name = "inverse_filter_%d.wav" % (n,)
    ir_name = "impulse_response_%d.wav" % (n,)

    wavfile.write(f"{sweep_name}", sample_rate, sweep)
    wavfile.write(f"{inverse_filter_name}", sample_rate, inverse)
    wavfile.write(f"{ir_name}", sample_rate, ir)
"""
    if n != number_of_repetitions:
        time.sleep(sweep_duration)
"""

ir_list = []
for file_name in glob.glob("impulse_response_*.*"):
    sample_rate, data = wavfile.read(file_name)
    ir_list.append(data)

mean_ir = np.mean(ir_list)
wavfile.write("mean_ir.wav", sample_rate, ir)

datadir = Path("/Users/ettorecarlessi/Documents/PyCharm/Projects/rev_room")
file_ir = datadir / "mean_ir.wav"
f = open(file_ir)

t60 = t60_impulse(file_ir, third(100, 5000), rt='t30')
# print(t60)

print(sabine_absorption_area(t60, 300))

"""
def save_file():
    file = filedialog.asksaveasfile(initialdir="/Users/ettorecarlessi/Documents/PyCharm/Projects/rev_room",
                                    defaultextension=".txt",
                                    filetypes=[
                                        ("Text file", ".txt"),
                                        ("HTML file", ".html"),
                                        ("All files", ".*")
                                    ])
    file_text = str(text.get(1.0, END))
    file.write(file_text)
    file.close()


window = Tk()
button = Button(text="Save", command=save_file)
button.pack()
text = Text(window)
text.pack()
window.mainLoop()
"""

# fare le sweep, calcolare ir per ogni singola sweep, media IR, filtro bande di terzi di ottava + integrale di schroedern (https://python-acoustics.github.io/python-acoustics/_modules/acoustics/room.html)(così recupero il segnale), calcolo t60(interpolazione linare sul decadimenot in dB, quindi ho t riv prima e dopo) , formula di sabin -> quindi assorbimento = 0,161 * V / t60

# chiedere se vengono usate comuqnue formule sabine in camera per calcoli. I filtri devono essere a norma ISO
# aggiungere sf play e record
