from tinygrad import Tensor, nn
from tinygrad.nn.state import get_state_dict, load_state_dict

def copy_weights(src, dst):
    states = get_state_dict(src)
    load_state_dict(dst, states, verbose=False)
