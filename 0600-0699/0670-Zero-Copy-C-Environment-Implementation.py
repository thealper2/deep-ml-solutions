import numpy as np

def zero_copy_env_simulation(num_envs: int, obs_dim: int, step_data: list, commands: list) -> dict:
    """
    Simulate a vectorized C environment with zero-copy buffer management.

    Args:
        num_envs: Number of parallel environments.
        obs_dim: Dimension of each observation vector.
        step_data: List of (obs_2d, rewards, dones) tuples for each timestep.
        commands: List of command tuples to execute.

    Returns:
        Dictionary with keys: 'reads', 'alias_checks', 'total_copies',
        'total_views', 'total_bytes_saved', 'buffer_reallocs'.
    """
    obs_buffer = np.zeros((num_envs, obs_dim), dtype=np.float64)
    reward_buffer = np.zeros(num_envs, dtype=np.float64)
    done_buffer = np.zeros(num_envs, dtype=np.int8)

    step_idx = 0
    stored_views = {}
    stored_copies = {}
    reads = []
    alias_checks = []
    total_views = 0
    total_copies = 0
    total_bytes_saved = 0
    buffer_reallocs = 0

    def step_env():
        nonlocal step_idx
        if step_idx >= len(step_data):
            return

        obs_2d, rewards, dones = step_data[step_idx]
        step_idx += 1

        obs_buffer[:] = np.array(obs_2d, dtype=np.float64)
        reward_buffer[:] = np.array(rewards, dtype=np.float64)
        done_buffer[:] = np.array(dones, dtype=np.int8)

    def auto_reset():
        done_mask = done_buffer == 1
        obs_buffer[done_mask] = 0.0
        reward_buffer[done_mask] = 0.0
    
    def store_view(name):
        nonlocal total_views, total_bytes_saved
        stored_views[name] = obs_buffer
        total_views += 1
        total_bytes_saved += num_envs * obs_dim * 8
    
    def store_copy(name):
        nonlocal total_copies
        stored_copies[name] = obs_buffer.copy()
        total_copies += 1
    
    def read(name):
        if name in stored_views:
            reads.append(stored_views[name].tolist())
        elif name in stored_copies:
            reads.append(stored_copies[name].tolist())
        else:
            reads.append([])
    
    def read_buffer():
        reads.append(obs_buffer.tolist())
    
    def check_alias(name):
        if name in stored_views:
            alias_checks.append(stored_views[name].base is obs_buffer.base)
        elif name in stored_copies:
            alias_checks.append(False)
        else:
            alias_checks.append(False)
    
    for cmd in commands:
        if cmd[0] == "step":
            step_env()
        elif cmd[0] == "auto_reset":
            auto_reset()
        elif cmd[0] == "store_view":
            store_view(cmd[1])
        elif cmd[0] == "store_copy":
            store_copy(cmd[1])
        elif cmd[0] == "read":
            read(cmd[1])
        elif cmd[0] == "read_buffer":
            read_buffer()
        elif cmd[0] == "check_alias":
            check_alias(cmd[1])
    
    return {
        "reads": reads,
        "alias_checks": alias_checks,
        "total_copies": total_copies,
        "total_views": total_views,
        "total_bytes_saved": total_bytes_saved,
        "buffer_reallocs": buffer_reallocs
    }
