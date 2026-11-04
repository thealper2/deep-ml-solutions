import numpy as np
import heapq


def push_priority(queue, q_table, state, action, reward, next_state, gamma, theta):
    target = reward + gamma * float(q_table[next_state].max())
    priority = abs(target - float(q_table[state][action]))
    if priority > theta:
        heapq.heappush(queue, (-float(priority), tuple(state), int(action)))

    return queue