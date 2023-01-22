from sympy.discrete.transforms import *
from sympy.discrete.convolutions import *
import numpy as np
import py3gpp
import matplotlib.pyplot as plt
from py3gpp.helper import _calc_m_seq
from numpy.fft import fft, ifft

def periodic_corr(x, y):
    """Periodic correlation, implemented using the FFT.

    x and y must be real sequences with the same length.
    """
    return ifft(fft(x) * fft(y).conj()).real

def _gold(N_id_1, N_id_2):
    N = 7
    c = [1, 0, 0, 0, 0, 0, 0]
    taps_0 = [0, 4]
    taps_1 = [0, 1]
    mseq_0 = _calc_m_seq(N, c, taps_0)
    mseq_1 = _calc_m_seq(N, c, taps_1)
    m_0 = 15 * int((N_id_1 / 112)) + 5 * N_id_2
    m_1 = N_id_1 % 112
    d_SSS = (1 - 2 * np.roll(mseq_0, -m_0)) * (1 - 2 * np.roll(mseq_1, -m_1))
    return d_SSS


pss = py3gpp.nrPSS(0)
pss_shifted = py3gpp.nrPSS(1)
N_id = 69
N_id_2 = N_id % 3
N_id_1 = (N_id - N_id_2) // 3
print(f'N_id_1 = {N_id_1}  N_id_2 = {N_id_2}')
sss = py3gpp.nrSSS(N_id)

N = 7
c = [1, 0, 0, 0, 0, 0, 0]
taps_0 = [0, 4]
taps_1 = [0, 1]
m_0 = 15 * int((N_id_1 / 112)) + 5 * N_id_2
m_1 = N_id_1 % 112
x_0 = _calc_m_seq(N, c, taps_0)
x_0_bpsk = 1 - 2*np.roll(x_0, -m_0)

sss_minus = sss * x_0_bpsk

x_1 = _calc_m_seq(N, c, taps_1)
x_1_bpsk = 1 - 2*np.roll(x_1, 0)


#plt.plot(convolution_fwht(((sss_minus+1)//2).tolist(), (x_1).tolist()))
#plt.plot((ifwht(np.array(fwht((sss_minus.tolist())) * np.array(fwht(x_1_bpsk.tolist()))))))
plt.plot(np.array(fwht((-(sss_minus-1)//2).tolist())) * x_1)

xcorr = periodic_corr(sss_minus, x_1_bpsk)
#plt.plot(xcorr)
detected_N_id_1 = 127 - np.argmax(xcorr)
print(f'detected N_id = {3*detected_N_id_1 + N_id_2}')
plt.show()