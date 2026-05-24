import unittest
import numpy as np
import sys
import os

# Add the python directory to the path so we can import the reward modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from reward.spectral import SpectralSmoothnessReward
from reward.lyapunov import LyapunovReward

class TestRewards(unittest.TestCase):
    def test_lyapunov_reward_moving_closer(self):
        reward_fn = LyapunovReward()
        
        # Initial position
        ee_pos_1 = np.array([0.0, 0.0, 0.0])
        target_pos = np.array([1.0, 0.0, 0.0])
        
        # First call sets prev_error
        r1 = reward_fn.compute(ee_pos_1, target_pos)
        self.assertEqual(r1, 0.0)
        self.assertEqual(reward_fn.prev_error, 1.0)
        
        # Move closer: error decreases from 1.0 to 0.5
        ee_pos_2 = np.array([0.5, 0.0, 0.0])
        r2 = reward_fn.compute(ee_pos_2, target_pos)
        
        # New reward is (current_error - prev_error) = (0.5 - 1.0) = -0.5
        self.assertAlmostEqual(r2, -0.5)
        
    def test_lyapunov_reward_moving_away(self):
        reward_fn = LyapunovReward()
        
        ee_pos_1 = np.array([0.5, 0.0, 0.0])
        target_pos = np.array([1.0, 0.0, 0.0])
        reward_fn.compute(ee_pos_1, target_pos) # prev_error = 0.5
        
        # Move away: error increases from 0.5 to 0.8
        ee_pos_2 = np.array([0.2, 0.0, 0.0])
        r2 = reward_fn.compute(ee_pos_2, target_pos)
        
        # New reward is (0.8 - 0.5) = 0.3
        self.assertAlmostEqual(r2, 0.3)

    def test_spectral_reward_no_jitter(self):
        # Constant signal (no high frequency power)
        reward_fn = SpectralSmoothnessReward(buffer_size=10, cutoff_ratio=0.5)
        for _ in range(10):
            reward_fn.push(np.array([1.0, 1.0, 1.0]))
        
        r = reward_fn.compute()
        # High freq power should be 0
        self.assertAlmostEqual(r, 0.0)

    def test_spectral_reward_high_jitter(self):
        # Alternating signal with offset (ensures low frequency power > 0)
        reward_fn = SpectralSmoothnessReward(buffer_size=10, cutoff_ratio=0.1)
        for i in range(10):
            val = 10.0 + (1.0 if i % 2 == 0 else -1.0)
            reward_fn.push(np.array([val, val, val]))
        
        r = reward_fn.compute()
        # Should be > 0 because of high frequency components
        self.assertGreater(r, 0.0)

if __name__ == "__main__":
    unittest.main()
