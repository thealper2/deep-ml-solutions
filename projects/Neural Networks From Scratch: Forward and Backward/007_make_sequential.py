def make_sequential(layers):
    """Compose protocol-honoring layers into one sequential model.

    Inputs:
      layers: list of layer dicts, each with
        forward(x) -> (y, cache),
        backward(dout, cache) -> (dx, grads_dict),
        params: dict of ndarrays (possibly empty).

    Returns a dict with:
      forward(x) -> (y, caches)
        y: final activation after applying every layer in order
        caches: opaque structure needed by backward
      backward(dout, caches) -> (dx, grads_list)
        dx: gradient w.r.t. the original input x
        grads_list: list of length len(layers); grads_list[i] is the
          grads_dict from layers[i] ({} for param-free layers)
      params: aggregated live view of all layer params, length len(layers),
        same order as layers (so in-place updates affect the model)
    """
    params = [layer['params'] for layer in layers]

    def forward(x):
      caches = []
      for layer in layers:
        x, cache = layer['forward'](x)
        caches.append(cache)

      return x, caches

    def backward(dout, caches):
      grads_list = []
      dx = dout
      for layer, cache in zip(reversed(layers), reversed(caches)):
        dx, grads = layer['backward'](dx, cache)
        grads_list.insert(0, grads)

      return dx, grads_list

    return {
      'params': params,
      'forward': forward,
      'backward': backward,
    }