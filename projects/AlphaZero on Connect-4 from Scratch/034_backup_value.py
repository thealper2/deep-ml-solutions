def backup_value(leaf, value):
    node = leaf
    sign = 1.0

    while node is not None:
        node['visit_count'] += 1
        node['value_sum'] += sign * value
        sign = -sign
        node['visits'] = node['visit_count']
        node = node['parent']
