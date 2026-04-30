def gym_api_convert(data, call_type: str, direction: str):
    """
    Convert between old Gym and new Gymnasium API output formats.

    Args:
        data: The output from reset() or step() in the source format.
        call_type: 'reset' or 'step'
        direction: 'old_to_new' or 'new_to_old'

    Returns:
        The converted output in the target API format.
    """
    if call_type == 'reset':
        if direction == 'old_to_new':
            observation = data
            info = {}
            return (observation, info)
        else:
            observation, info = data
            return observation

    elif call_type == 'step':
        if direction == 'old_to_new':
            observation, reward, done, info = data
            terminated = done and not info.get('TimeLimit.truncated', False)
            truncated = done and info.get('TimeLimit.truncated', False)
            new_info = info.copy()
            new_info.pop('TimeLimit.truncated', None)
            return (observation, reward, terminated, truncated, new_info)
        else:
            observation, reward, terminated, truncated, info = data
            done = terminated or truncated
            new_info = info.copy()
            if truncated:
                new_info['TimeLimit.truncated'] = True
            else:
                new_info.pop('TimeLimit.truncated', None)

            return (observation, reward, done, new_info)
    else:
        raise ValueError('Invalid call_type.')