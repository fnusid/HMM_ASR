import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import os
import pdb
import matplotlib.pyplot as plt
import scipy
import librosa.display
from scipy.signal import get_window
from mfcc import _WindowingMFCC
class compute_delta():
    def __init__(self) -> None:
        pass
    def normalize_features(self, features):
        mean = np.mean(features, axis=1, keepdims=True)
        std = np.std(features, axis=1, keepdims=True)
        return (features - mean) / (std + 1e-10)
    def _delta(self,audio, sr, M=2):
        audio=audio
        sr = sr
        mfcc_matrix = _WindowingMFCC().run(audio, sr)

        N, T = mfcc_matrix.shape[0], mfcc_matrix.shape[1]
        delta = np.zeros((N, T-(2*M)))
        M_vec = [-M,-M+1, M, M+1, M+2 ]
        for n in range(N):
            nr, dr = 0., 0.
            for t in range(2,T-2):
                feat_vec = mfcc_matrix[n,t-2:t+3]
                try:
                    nr = np.dot(feat_vec, M_vec)
                except ValueError:
                    breakpoint()
                dr = np.sum(np.square(M_vec))
                delta[n,t-2] = (nr/dr)
        concatenated_vector = np.concatenate([mfcc_matrix[:,2:-2], delta], axis=0)






        # T, coefficients = mfcc_matrix.shape[0], mfcc_matrix.shape[1]

        # delta = np.zeros((T, coefficients))

        # for frame_no in range(T):
        #     numerator, denominator = 0., 0.

        #     for m in range(M):
        #         start_coefficients, end_coefficients = mfcc_matrix[0, :], mfcc_matrix[T - 1, :]
        #         if frame_no - m >= 0:
        #             start_coefficients = mfcc_matrix[frame_no - m, :]
        #         if frame_no + m < T:
        #             end_coefficients = mfcc_matrix[frame_no + m, :]

        #         numerator += m * (end_coefficients - start_coefficients)
        #         denominator += m ** 2

        #     denominator *= 2
        #     delta[frame_no, :] = numerator / denominator
        normalized_vector = self.normalize_features(concatenated_vector)
        return normalized_vector
    
    def _forward(self,audio, sr):
        audio, sr = audio, sr
        return self._delta(audio, sr, M=2)

# delta_comp = compute_delta()._forward()


