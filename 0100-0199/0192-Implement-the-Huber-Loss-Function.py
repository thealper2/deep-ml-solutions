import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
	"""
	Compute the Huber Loss between true and predicted values.

	Args:
		y_true (float | list[float]): Ground truth values
		y_pred (float | list[float]): Predicted values
		delta (float): Transition threshold between MSE and MAE behavior

	Returns:
		float: Average Huber loss
	"""
	y_true = np.array(y_true)
	y_pred = np.array(y_pred)
	error = y_true - y_pred
	is_small_error = np.abs(error) <= delta
	squared_loss = 0.5 * np.square(error)
	linear_loss = delta * (np.abs(error) - 0.5 * delta)
	loss = np.where(is_small_error, squared_loss, linear_loss)
	return float(np.mean(loss))