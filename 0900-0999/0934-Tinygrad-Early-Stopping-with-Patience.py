class EarlyStopping:
    def __init__(self, patience, mode='min'):
        self.patience = patience
        self.mode = mode
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
    
    def step(self, metric):
        if self.mode == 'min':
            if metric < self.best:
                self.best = metric
                self.counter = 0
            else:
                self.counter += 1

        else:
            if metric > self.best:
                self.best = metric
                self.counter = 0
            else:
                self.counter += 1

        return self.counter >= self.patience