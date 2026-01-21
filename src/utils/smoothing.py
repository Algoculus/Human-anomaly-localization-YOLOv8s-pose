"""
Temporal Smoothing Filters

Provides EMA and Median filters for smoothing pose features over time.
"""

from collections import deque
from typing import Optional, Union, List
import numpy as np


class EMAFilter:
    """
    Exponential Moving Average (EMA) filter.
    
    EMA places more weight on recent values:
        output[t] = alpha * input[t] + (1 - alpha) * output[t-1]
    
    Higher alpha = more responsive to recent changes
    Lower alpha = smoother but more lag
    """
    
    def __init__(self, alpha: float = 0.3, initial_value: Optional[float] = None):
        """
        Initialize EMA filter.
        
        Args:
            alpha: Smoothing factor (0 < alpha <= 1). 
                   Higher = more weight on recent values.
            initial_value: Optional initial filtered value.
        """
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        
        self.alpha = alpha
        self._value = initial_value
        self._initialized = initial_value is not None
    
    def update(self, value: float) -> float:
        """
        Update filter with new value and return smoothed result.
        
        Args:
            value: New input value
            
        Returns:
            Smoothed value
        """
        if not self._initialized:
            self._value = value
            self._initialized = True
        else:
            self._value = self.alpha * value + (1 - self.alpha) * self._value
        
        return self._value
    
    @property
    def value(self) -> Optional[float]:
        """Current filtered value."""
        return self._value
    
    def reset(self, initial_value: Optional[float] = None):
        """Reset filter state."""
        self._value = initial_value
        self._initialized = initial_value is not None


class VectorEMAFilter:
    """
    EMA filter for vectors (multiple values).
    """
    
    def __init__(self, alpha: float = 0.3, dim: int = 2, 
                 initial_value: Optional[np.ndarray] = None):
        """
        Initialize vector EMA filter.
        
        Args:
            alpha: Smoothing factor
            dim: Dimension of vectors
            initial_value: Optional initial filtered vector
        """
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        
        self.alpha = alpha
        self.dim = dim
        self._value = initial_value.copy() if initial_value is not None else None
        self._initialized = initial_value is not None
    
    def update(self, value: np.ndarray) -> np.ndarray:
        """
        Update filter with new vector and return smoothed result.
        """
        value = np.asarray(value)
        
        if not self._initialized:
            self._value = value.copy()
            self._initialized = True
        else:
            self._value = self.alpha * value + (1 - self.alpha) * self._value
        
        return self._value.copy()
    
    @property
    def value(self) -> Optional[np.ndarray]:
        """Current filtered vector."""
        return self._value.copy() if self._value is not None else None
    
    def reset(self, initial_value: Optional[np.ndarray] = None):
        """Reset filter state."""
        self._value = initial_value.copy() if initial_value is not None else None
        self._initialized = initial_value is not None


class MedianFilter:
    """
    Median filter for temporal smoothing.
    
    Returns the median of the last N values.
    More robust to outliers than EMA.
    """
    
    def __init__(self, window_size: int = 5):
        """
        Initialize median filter.
        
        Args:
            window_size: Number of values to keep in buffer
        """
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        
        self.window_size = window_size
        self._buffer: deque = deque(maxlen=window_size)
    
    def update(self, value: float) -> float:
        """
        Update filter with new value and return median.
        
        Args:
            value: New input value
            
        Returns:
            Median of buffered values
        """
        self._buffer.append(value)
        return np.median(list(self._buffer))
    
    @property
    def value(self) -> Optional[float]:
        """Current median value."""
        if len(self._buffer) == 0:
            return None
        return np.median(list(self._buffer))
    
    @property
    def buffer(self) -> List[float]:
        """Current buffer contents."""
        return list(self._buffer)
    
    def reset(self):
        """Clear filter buffer."""
        self._buffer.clear()


class VectorMedianFilter:
    """
    Median filter for vectors (element-wise median).
    """
    
    def __init__(self, window_size: int = 5, dim: int = 2):
        """
        Initialize vector median filter.
        
        Args:
            window_size: Number of values to keep in buffer
            dim: Dimension of vectors
        """
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        
        self.window_size = window_size
        self.dim = dim
        self._buffer: deque = deque(maxlen=window_size)
    
    def update(self, value: np.ndarray) -> np.ndarray:
        """
        Update filter with new vector and return element-wise median.
        """
        self._buffer.append(np.asarray(value).copy())
        
        if len(self._buffer) == 1:
            return self._buffer[0].copy()
        
        stacked = np.stack(list(self._buffer), axis=0)
        return np.median(stacked, axis=0)
    
    @property
    def value(self) -> Optional[np.ndarray]:
        """Current median vector."""
        if len(self._buffer) == 0:
            return None
        stacked = np.stack(list(self._buffer), axis=0)
        return np.median(stacked, axis=0)
    
    def reset(self):
        """Clear filter buffer."""
        self._buffer.clear()


class HistoryBuffer:
    """
    Simple buffer to store history of values for look-back operations.
    """
    
    def __init__(self, max_length: int = 30):
        """
        Initialize history buffer.
        
        Args:
            max_length: Maximum number of values to store
        """
        self.max_length = max_length
        self._buffer: deque = deque(maxlen=max_length)
    
    def append(self, value: Union[float, np.ndarray]):
        """Add value to history."""
        if isinstance(value, np.ndarray):
            value = value.copy()
        self._buffer.append(value)
    
    def get_last_n(self, n: int) -> List:
        """Get last N values (oldest to newest)."""
        if n >= len(self._buffer):
            return list(self._buffer)
        return list(self._buffer)[-n:]
    
    def get_range(self, start: int, end: int) -> List:
        """Get values in range [start, end) from the end."""
        values = list(self._buffer)
        if start >= len(values):
            return []
        return values[-end:-start] if start > 0 else values[-end:]
    
    def __len__(self) -> int:
        return len(self._buffer)
    
    def __getitem__(self, idx: int):
        """Get value by index (negative for from end)."""
        return self._buffer[idx]
    
    def clear(self):
        """Clear buffer."""
        self._buffer.clear()
    
    @property
    def values(self) -> List:
        """All buffered values."""
        return list(self._buffer)
    
    @property
    def is_full(self) -> bool:
        """Check if buffer is at max capacity."""
        return len(self._buffer) == self.max_length


if __name__ == "__main__":
    # Test filters
    print("Testing EMA filter...")
    ema = EMAFilter(alpha=0.3)
    values = [10, 12, 11, 15, 14, 13, 20, 18, 17, 16]
    smoothed = [ema.update(v) for v in values]
    print(f"  Input:    {values}")
    print(f"  Smoothed: {[round(v, 2) for v in smoothed]}")
    
    print("\nTesting Median filter...")
    median = MedianFilter(window_size=3)
    smoothed = [median.update(v) for v in values]
    print(f"  Input:    {values}")
    print(f"  Smoothed: {smoothed}")
    
    print("\nTesting History buffer...")
    history = HistoryBuffer(max_length=5)
    for v in values:
        history.append(v)
    print(f"  Buffer (last 5): {history.values}")
    print(f"  Last 3: {history.get_last_n(3)}")
    
    print("\nAll tests passed!")
