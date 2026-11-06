import sys
import os
import re
import math
import itertools
import shutil
import subprocess
import tempfile
import hashlib
import ctypes
import pathlib
import urllib.request
from math import prod
from collections import namedtuple
import numpy as np


MNIST_NPZ = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


def prepare_mnist(n_train, n_test, folder=None):
    try:
        import dml_data
        ds = dml_data.load("mnist")
        xtr, ytr, xte, yte = ds.x_train, ds.y_train, ds.x_test, ds.y_test
    except Exception:
        path = os.path.join(tempfile.gettempdir(), "mnist.npz")
        if not os.path.exists(path):
            urllib.request.urlretrieve(MNIST_NPZ, path)
        with np.load(path) as f:
            xtr, ytr, xte, yte = f["x_train"], f["y_train"], f["x_test"], f["y_test"]
    folder = folder or tempfile.mkdtemp(prefix="uir_mnist_")
    eye = np.eye(10, dtype=np.float32)
    arrays = {"x_train": xtr[:n_train].reshape(n_train, 784) / np.float32(255), "y_train": eye[ytr[:n_train]],
              "x_test": xte[:n_test].reshape(n_test, 784) / np.float32(255), "y_test": eye[yte[:n_test]]}
    files = {}
    for name, arr in arrays.items():
        data = np.ascontiguousarray(arr, dtype="<f4").tobytes()
        path = os.path.join(folder, name + ".bin")
        with open(path, "wb") as f:
            f.write(data)
        files[name] = (pathlib.Path(path).as_uri(), hashlib.sha256(data).hexdigest())
    return files


def mnist_program(n_train, n_test, files, hidden=32, batch=16, lr=0.1, epochs=1, key=(7, 11)):
    H, b = hidden, batch
    n1 = 784 * H
    P = n1 + H * 10
    batches = n_train // b
    steps = epochs * batches
    s1 = repr(0.08 / 2 ** 31)
    s2 = repr(0.3 / 2 ** 31)
    inv_b = repr(1.0 / b)
    neg_lr = repr(-float(lr))

    def buf(name, shape, slot):
        url, sha = files[name]
        return f"buffer [{shape[0]} {shape[1]}] : dtype=f32 slot={slot} url={url} sha256={sha}"

    def matmul(a, w, m, k, n):
        return (f"reduce (permute (mul (expand (reshape {a} [{m} {k} 1]) [{m} {k} {n}]) "
                f"(expand (reshape {w} [1 {k} {n}]) [{m} {k} {n}])) [1 0 2]) : op=add pop=1")

    lines = [
        f"%w = buffer {P} : dtype=f32 slot=0",
        f"%xtr = {buf('x_train', (n_train, 784), 1)}",
        f"%ytr = {buf('y_train', (n_train, 10), 2)}",
        f"%xte = {buf('x_test', (n_test, 784), 3)}",
        f"%yte = {buf('y_test', (n_test, 10), 4)}",
        "%correct = buffer 1 : dtype=i32 slot=5",
        f"%i = range {P} : slot=0",
        f"%rnd = index (call %threefry {key[0]} {key[1]} %i 0) 0",
        f"%val = mul (cast %rnd : f32) (where (cmplt %i {n1}) {s1} {s2})",
        "%init = end (store (index %w %i) %val) %i",
        "%w0 = after %w %init",
        f"%step = range {steps} : slot=1",
        f"%start = mul (mod %step {batches}) {b}",
        "%wv = load (after %w0 %step)",
        f"%w1 = reshape (shrink %wv 0 [{n1}]) [784 {H}]",
        f"%w2 = reshape (shrink %wv {n1} [{H * 10}]) [{H} 10]",
        f"%x = load (shrink %xtr %start [{b}])",
        f"%y = load (shrink %ytr %start [{b}])",
        f"%h = max ({matmul('%x', '%w1', b, 784, H)}) 0.0",
        f"%logits = {matmul('%h', '%w2', b, H, 10)}",
        "%err = add %logits (mul %y -1.0)",
        f"%loss = mul (reduce (mul %err %err) : op=add pop=2) {inv_b}",
        f"%train = end (store %w0 (add %wv (mul (stage (gradient %loss %wv)) {neg_lr}))) %step",
        "%wt = load (after %w %train)",
        f"%v1 = reshape (shrink %wt 0 [{n1}]) [784 {H}]",
        f"%v2 = reshape (shrink %wt {n1} [{H * 10}]) [{H} 10]",
        "%xt = load %xte",
        "%yt = load %yte",
        f"%ht = max ({matmul('%xt', '%v1', n_test, 784, H)}) 0.0",
        f"%lg = {matmul('%ht', '%v2', n_test, H, 10)}",
        "%tl = reduce (permute (mul %lg %yt) [1 0]) : op=add pop=1",
        "%om = reduce (permute (where (cmplt %yt 0.5) %lg -inf) [1 0]) : op=max pop=1",
        "%acc = reduce (cast (cmplt %om %tl) : i32) : op=add pop=1",
        "%result = after %correct (store (index %correct 0) %acc)",
    ]
    return threefry_block() + "\n".join(lines) + "\n"