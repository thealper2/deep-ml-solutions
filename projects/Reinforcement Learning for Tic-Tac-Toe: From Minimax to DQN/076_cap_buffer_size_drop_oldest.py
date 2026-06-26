def cap_buffer_size_drop_oldest(buffer):
    """Drop oldest transitions until len(buffer['data']) <= buffer['capacity']."""
    while len(buffer['data']) > buffer['capacity']:
        buffer['data'].pop(0)

    return buffer
