import numpy as np
from collections import deque

class SpectralSmoothnessReward:
    def __init__(self, buffer_size=64, cutoff_ratio=0.15):
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.cutoff_ratio = cutoff_ratio

    def push(self, ee_pos: np.ndarray):
        self.buffer.append(ee_pos)

    def compute(self) -> float:
        if len(self.buffer) < self.buffer_size:
            return 0.0
        
        data = np.array(self.buffer)
        total_high_freq_power = 0.0
        total_low_freq_power = 0.0
        
        for i in range(3):
            fft_vals = np.fft.rfft(data[:, i])
            power = np.abs(fft_vals) ** 2
            
            cutoff_idx = int(len(power) * self.cutoff_ratio)
            if cutoff_idx == 0:
                cutoff_idx = 1
                
            low_freq_power = np.sum(power[:cutoff_idx])
            high_freq_power = np.sum(power[cutoff_idx:])
            
            total_low_freq_power += low_freq_power
            total_high_freq_power += high_freq_power
            
        if total_low_freq_power == 0:
            return 0.0
            
        return total_high_freq_power / total_low_freq_power
