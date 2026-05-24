import numpy as np

class LyapunovReward:
    def __init__(self):
        self.prev_error = None

    def reset(self):
        self.prev_error = None

    def compute(self, ee_pos: np.ndarray, target_pos: np.ndarray) -> float:
        current_error = np.linalg.norm(ee_pos - target_pos)
        
        if self.prev_error is None:
            self.prev_error = current_error
            return 0.0
            
        reward = current_error - self.prev_error
        self.prev_error = current_error
        
        return reward
