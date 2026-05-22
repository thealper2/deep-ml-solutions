import numpy as np

def find_treasure(start_x: float) -> float:
  """
  Find the x-coordinate where f(x) = x^4 - 3x^3 + 2 is minimized.

  Returns:
    float: The x-coordinate of the minimum point.
  """
  def f(x):
    return x**4 - 3*x**3 + 2
    
  x_vals = np.linspace(-5, 5, 10000)
  y_vals = f(x_vals)
  min_index = np.argmin(y_vals)   
  return float(x_vals[min_index])