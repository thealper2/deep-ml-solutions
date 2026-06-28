class LazyBuffer:
    def __init__(self, np_array):
        self._np = np.array(np_array)
        self.shape = self._np.shape
        self.dtype = self._np.dtype        
