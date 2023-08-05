#!/usr/bin/env python3

#
# Modulation testing
# Testing if the phase compensation helps capture partial bandwidth
#
# Author: Sylvain Munaut <tnt@246tNt.com>
#

import math

import numpy as np


FFT_LEN = 4096
CP_LEN  =  128

Fc = 1e9		# Center freq modulation
Fs = 122.88e6	# Sample Rate
C_OFS = 10		# How many carries to use as offset

PHASE_COMP = True


# Generate symbols with a zone of fixed phase
# constants in the upper part of the spectrum
# Expect 0.628 radian phase difference between symbols

def sym_gen(p):
	sym = np.zeros(FFT_LEN, dtype=np.complex64)
	s = 5 * FFT_LEN // 8
	e = 6 * FFT_LEN // 8
	sym[s:e] = np.exp(1j * p)
	return sym


# OFDM symbol modulation / demodulation
# (mod is done at Fs, demod is assumed to be done at half rate)

def sym_modulate(s, idx=0, f=None):
	# Phase compensation
	if PHASE_COMP and (f is not None):
		pc_t = idx * (FFT_LEN + CP_LEN) / Fs
		pc_a = - f * pc_t * 2 * math.pi
		pc = np.exp(1j*pc_a)
		s = s * pc

	# Modulate and add CP
	m = np.fft.ifft(np.fft.ifftshift(s))
	return np.concatenate([m[-CP_LEN:], m])


def sym_demodulate(m, idx=0, f=None):
	# Removes CP and demod
	m = m[CP_LEN//2:]
	s = np.fft.fftshift(np.fft.fft(m))

	# Phase compensation
	if PHASE_COMP and (f is not None):
		pc_t = idx * (FFT_LEN + CP_LEN) / Fs
		pc_a = f * pc_t * 2 * math.pi
		pc = np.exp(1j*pc_a)
		s = s * pc

	return s


# Up/Down conversion to "RF"
# (assuming Fs sampling rate)

def upconvert(m, f):
	ang = 2.0 * math.pi * f / Fs
	return m * np.exp(1j *  ang * np.arange(len(m)))

def downconvert(m, f):
	ang = 2.0 * math.pi * f / Fs
	return m * np.exp(1j * -ang * np.arange(len(m)))



def main():
	# Mod
	# ---

	# Modulate symbols with some constant 2pi/10 phase offset between them
	d = []

	for i in range(10):
		s = sym_gen(i * (2 * math.pi / 10))
		m = sym_modulate(s, i, Fc)
		d.append(m)

	# Generate "RF"
	mod_tx = np.concatenate(d)
	sig_tx = upconvert(mod_tx, Fc)


	# Demod
	# -----

	# "Tune" to only the upper part and also offset by a few sub carriers
	# (if you tune by exacty 1/4 of Fs, phase comp isn't needed ...)
	f_ofs = (Fs / FFT_LEN) * ((FFT_LEN / 4) - C_OFS)
	mod_rx = downconvert(sig_tx, Fc + f_ofs)

	# Down sample to simulate RX at a lower bandwidth
	mod_rx = mod_rx[::2]

	# Demod each symbol
	r = []
	ra = []
	for i in range(10):
		s_len = (FFT_LEN + CP_LEN) // 2
		m = mod_rx[i*s_len:(i+1)*s_len]
		s = sym_demodulate(m, i, Fc + f_ofs)
		r.append(s)
		ra.append( sum(s[512+C_OFS:1024+C_OFS]) / 512 )

	# Print the average phase offset between symbols (expected to be 2pi/10)
	ad = [np.angle(b*a.conj()) for a,b in zip(ra, ra[1:])]
	print(f"Actual: {sum(ad)/len(ad):f},  Expected {2*math.pi/10:f}" )


if __name__ == '__main__':
	main()