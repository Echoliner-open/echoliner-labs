"""Signal processing utilities for speech translation pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

__all__ = ["SpeechFeatureExtractor", "griffin_lim"]


@dataclass
class SpeechFeatureExtractor:
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 160
    n_mels: int = 80
    preemphasis: float = 0.97

    def mel_filter_bank(self) -> NDArray[np.float64]:
        low_freq = 0
        high_freq = self.sample_rate / 2
        mel_low = self._hz_to_mel(low_freq)
        mel_high = self._hz_to_mel(high_freq)
        mel_points = np.linspace(mel_low, mel_high, self.n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)
        bins = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)
        filter_bank = np.zeros((self.n_mels, self.n_fft // 2 + 1))
        for i in range(1, self.n_mels + 1):
            left = bins[i - 1]
            center = bins[i]
            right = bins[i + 1]
            for j in range(left, center):
                filter_bank[i - 1, j] = (j - left) / (center - left + 1e-12)
            for j in range(center, right):
                filter_bank[i - 1, j] = (right - j) / (right - center + 1e-12)
        return filter_bank

    def _hz_to_mel(self, hz: float) -> float:
        return 2595 * np.log10(1 + hz / 700)

    def _mel_to_hz(self, mel: NDArray[np.float64]) -> NDArray[np.float64]:
        return 700 * (10 ** (mel / 2595) - 1)

    def stft(self, waveform: NDArray[np.float64]) -> NDArray[np.complex128]:
        frames = self._frame(waveform)
        window = np.hanning(self.n_fft)
        return np.fft.rfft(window * frames, n=self.n_fft)

    def _frame(self, waveform: NDArray[np.float64]) -> NDArray[np.float64]:
        padded = np.pad(waveform, (self.n_fft // 2, self.n_fft // 2), mode="reflect")
        frame_count = 1 + (len(padded) - self.n_fft) // self.hop_length
        frames = np.zeros((frame_count, self.n_fft))
        for i in range(frame_count):
            start = i * self.hop_length
            frames[i] = padded[start : start + self.n_fft]
        return frames

    def spectrogram(self, waveform: NDArray[np.float64]) -> NDArray[np.float64]:
        preemphasized = np.append(waveform[0], waveform[1:] - self.preemphasis * waveform[:-1])
        stft_matrix = self.stft(preemphasized)
        magnitude = np.abs(stft_matrix)
        return magnitude ** 2

    def mel_spectrogram(self, waveform: NDArray[np.float64]) -> NDArray[np.float64]:
        spec = self.spectrogram(waveform)
        mel_basis = self.mel_filter_bank()
        mel_spec = np.dot(mel_basis, spec.T)
        return np.log10(np.maximum(mel_spec, 1e-10))

    def mfcc(self, waveform: NDArray[np.float64], n_coeffs: int = 13) -> NDArray[np.float64]:
        mel_spec = self.mel_spectrogram(waveform)
        ceps = self._dct(mel_spec)
        return ceps[:n_coeffs]

    def _dct(self, matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        n = matrix.shape[0]
        k = np.arange(n)[:, None]
        n_range = np.arange(n)
        transform = np.cos(np.pi / n * (n_range + 0.5) * k)
        transform[0] /= np.sqrt(2)
        scale = np.sqrt(2 / n)
        return scale * transform @ matrix


def griffin_lim(magnitude: NDArray[np.float64], n_iter: int = 60, hop_length: int = 160) -> NDArray[np.float64]:
    angle = np.exp(2j * np.pi * np.random.rand(*magnitude.shape))
    complex_spec = magnitude * angle
    for _ in range(n_iter):
        waveform = np.fft.irfft(complex_spec, axis=0)
        reconstructed = np.fft.rfft(waveform, axis=0)
        complex_spec = magnitude * np.exp(1j * np.angle(reconstructed))
    waveform = np.fft.irfft(complex_spec, axis=0)
    return waveform.flatten()
