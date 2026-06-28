def argsort(values):
    return [i[0] for i in sorted(enumerate(values), key=lambda x: x[1])]
