import jax

def split_prng_key(key, num):
    return jax.random.split(key, num=num)
