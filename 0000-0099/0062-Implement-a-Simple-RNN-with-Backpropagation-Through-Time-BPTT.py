
import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        """Initializes the RNN with random weights and zero biases."""
        self.hidden_size = hidden_size
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))

    def forward(self, x):
        """Forward pass through the RNN for a given sequence of inputs."""
        h = np.zeros((self.hidden_size, 1))
        self.hidden_states = {-1: h}
        self.outputs = {}
        outputs = []
        for t in range(len(x)):
            x_t = x[t].reshape(-1, 1)
            h = np.tanh(self.W_xh @ x_t + self.W_hh @ self.hidden_states[t - 1] + self.b_h)
            y_t = self.W_hy @ h + self.b_y
            self.hidden_states[t] = h
            self.outputs[t] = y_t
            outputs.append(y_t)
			
        return np.array(outputs)

    def backward(self, x, y, learning_rate):
        """Backpropagation through time to adjust weights based on error gradient."""
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        dh_next = np.zeros((self.hidden_size, 1))

        for t in reversed(range(len(x))):
            y_t = y[t].reshape(-1, 1)
            dy = (self.outputs[t] - y_t)
            dW_hy += dy @ self.hidden_states[t].T
            db_y += dy

            dh = self.W_hy.T @ dy + dh_next
            dh_raw = (1 - self.hidden_states[t] ** 2) * dh
            db_h += dh_raw
            dW_xh += dh_raw @ x[t].reshape(-1, 1).T
            dW_hh += dh_raw @ self.hidden_states[t - 1].T
            dh_next = self.W_hh.T @ dh_raw

        self.W_xh -= learning_rate * dW_xh
        self.W_hh -= learning_rate * dW_hh
        self.W_hy -= learning_rate * dW_hy
        self.b_h -= learning_rate * db_h
        self.b_y -= learning_rate * db_y
