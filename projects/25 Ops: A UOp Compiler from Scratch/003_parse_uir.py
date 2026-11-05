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


_TOKEN = re.compile(r"\(|\)|\[|\]|\{|\}|[^\s()\[\]{}]+")
_INT = re.compile(r"-?(0x[0-9a-f]+|\d+)$")
_FLOAT = re.compile(r"-?((\d+\.\d*|\.\d+|\d+)(e[-+]?\d+)?|inf|nan)$")
_ALIASES = {"floordiv": "DIV", "floormod": "MOD"}
_TEXT_OPS = {"const", "param", "call", "stack", "index", "shrink", "cast", "recip", "trunc", "add", "mul", "max",
             "xor", "cmplt", "shl", "div", "mod", "where", "floordiv", "floormod", "buffer", "alloc", "load",
             "store", "after", "range", "end", "reduce", "expand", "stage", "reshape", "permute", "gradient",
             "detach", "pad"}
_STRING_ATTRS = ("url", "sha256", "name")
_AUTO_SLOT = itertools.count(1 << 20)


class Lit:
    __slots__ = ("v",)

    def __init__(self, v):
        self.v = v


def _literal(tok):
    t = tok.lower()
    if t in ("true", "false"):
        return t == "true"
    if _INT.match(t):
        return int(t, 0) if "0x" in t else int(t)
    if _FLOAT.match(t) and any(c in t for c in ".einfa"):
        return float(t)
    return None


def _lit_dtype(v):
    return "bool" if isinstance(v, bool) else "f32" if isinstance(v, float) else "i32"


def _to_uop(x, dtype=None, shape=()):
    if isinstance(x, UOp):
        return x
    if isinstance(x, list):
        d, s = _data(x)
        return UOp("STACK", [_to_uop(e, d, s) for e in x])
    v, d = x.v, dtype or _lit_dtype(x.v)
    if d == "bool" and not isinstance(v, bool):
        if isinstance(v, float):
            raise TypeError("a float literal cannot be a bool")
        v = bool(v)
    if d in ("i32", "f32") and isinstance(v, bool):
        raise TypeError("a bool literal cannot stand next to a number")
    c = const(v, d)
    if shape:
        c = UOp("EXPAND", (UOp("RESHAPE", (c,), (1,) * len(shape)),), tuple(shape))
    return c


def _data(operands):
    for a in operands:
        if isinstance(a, UOp):
            return a.dtype, a.shape
    lits = [a.v for a in operands if isinstance(a, Lit)]
    for a in operands:
        if isinstance(a, list):
            return _data(a)[0], (len(a),)
    if any(isinstance(v, float) for v in lits):
        return "f32", ()
    if lits and all(isinstance(v, bool) for v in lits):
        return "bool", ()
    return "i32", ()


def _ints(x):
    if isinstance(x, Lit):
        return (int(x.v),)
    if isinstance(x, UOp) and x.op == "CONST":
        return (int(x.arg),)
    if isinstance(x, list):
        return tuple(i for e in x for i in _ints(e))
    raise TypeError("expected a list of integers")


def _window(op, args):
    """SHRINK/PAD operands -> (x, begins, sizes). Forms: x [[b e]...] | x b size | x b0 b1 ... [s0 s1 ...]."""
    x = args[0]
    if len(args) == 2 and isinstance(args[1], list) and args[1] and isinstance(args[1][0], list):
        pairs = [(int(p[0].v), int(p[1].v)) for p in args[1]]
        begins = [const(b) for b, _ in pairs]
        if op == "SHRINK":
            sizes = tuple(e - b for b, e in pairs)
        else:
            sizes = tuple(b + n + a for (b, a), n in zip(pairs, x.shape))
        return x, begins, sizes
    if isinstance(args[-1], list):
        return x, [_to_uop(b, "i32") for b in args[1:-1]], _ints(args[-1])
    if len(args) == 3:
        return x, [_to_uop(args[1], "i32")], _ints(args[2])
    raise SyntaxError(f"bad {op.lower()} operands")


def _build(op, args, attrs):
    OP = _ALIASES.get(op, op.upper())

    def need(n):
        if len(args) != n:
            raise SyntaxError(f"{op} takes {n} operand(s), got {len(args)}")

    if OP in ("ADD", "MUL", "MAX", "XOR", "CMPLT", "SHL", "DIV", "MOD"):
        need(2)
        d, s = _data(args)
        return UOp(OP, [_to_uop(a, d, s) for a in args])
    if OP == "WHERE":
        need(3)
        d, s = _data(args[1:])
        if s == () and isinstance(args[0], UOp):
            s = args[0].shape
        return UOp("WHERE", [_to_uop(args[0], "bool", s), _to_uop(args[1], d, s), _to_uop(args[2], d, s)])
    if OP in ("RECIP", "TRUNC"):
        need(1)
        return UOp(OP, [_to_uop(args[0], "f32")])
    if OP == "CAST":
        need(1)
        return UOp("CAST", [_to_uop(args[0])], attrs["dtype"])
    if OP == "CONST":
        need(1)
        return _to_uop(args[0], attrs.get("dtype"))
    if OP == "PARAM":
        if args and _ints(args[0]) != ():
            raise TypeError("PARAMs are scalars")
        lo, hi = attrs.get("min"), attrs.get("max")
        return UOp("PARAM", (), ParamArg(int(attrs["slot"]), attrs["dtype"], lo, hi))
    if OP == "CALL":
        fn, rest = args[0], args[1:]
        params = _body_params(fn)
        return UOp("CALL", [fn] + [_to_uop(a, params[i].arg.dtype if i in params else None) for i, a in enumerate(rest)])
    if OP == "STACK":
        return _to_uop(list(args))
    if OP == "INDEX":
        return UOp("INDEX", [_to_uop(args[0])] + [_to_uop(a, "i32") for a in args[1:]])
    if OP in ("SHRINK", "PAD"):
        x, begins, sizes = _window(OP, args)
        return UOp(OP, [_to_uop(x)] + begins, tuple(sizes))
    if OP in ("BUFFER", "ALLOC"):
        need(1)
        return UOp(OP, (), BufArg(int(attrs["slot"]), attrs["dtype"], _ints(args[0]), attrs.get("url"), attrs.get("sha256")))
    if OP == "LOAD":
        need(1)
        return UOp("LOAD", [_to_uop(args[0])])
    if OP == "STORE":
        need(2)
        dst = _to_uop(args[0])
        return UOp("STORE", [dst, _to_uop(args[1], dst.dtype, dst.shape)])
    if OP == "AFTER":
        return UOp("AFTER", [_to_uop(a) for a in args])
    if OP == "RANGE":
        need(1)
        slot = attrs.get("slot", attrs.get("name"))
        return UOp("RANGE", [const(_ints(args[0])[0])], next(_AUTO_SLOT) if slot is None else int(slot))
    if OP == "END":
        need(2)
        return UOp("END", [_to_uop(a) for a in args])
    if OP == "REDUCE":
        need(1)
        return UOp("REDUCE", [_to_uop(args[0])], (str(attrs.get("op", "add")).upper(), int(attrs.get("pop", 1))))
    if OP in ("EXPAND", "RESHAPE", "PERMUTE"):
        need(2)
        return UOp(OP, [_to_uop(args[0])], _ints(args[1]))
    if OP in ("STAGE", "DETACH"):
        need(1)
        return UOp(OP, [_to_uop(args[0])])
    if OP == "GRADIENT":
        need(2)
        return UOp("GRADIENT", [_to_uop(a) for a in args])
    raise SyntaxError(f"unknown op {op!r}")


def _lookup(name, scopes):
    for sc in reversed(scopes):
        if name in sc:
            return sc[name]
    raise NameError(f"undefined name {name}")


def _parse_operand(toks, pos, scopes):
    if pos >= len(toks):
        raise SyntaxError("unexpected end of input")
    tok = toks[pos]
    if tok == "(":
        val, pos = _parse_expr(toks, pos + 1, scopes)
        if pos >= len(toks) or toks[pos] != ")":
            raise SyntaxError("expected ')'")
        return val, pos + 1
    if tok == "[":
        items = []
        pos += 1
        while pos < len(toks) and toks[pos] != "]":
            if toks[pos] in ("(", "["):
                v, pos = _parse_operand(toks, pos, scopes)
            elif toks[pos].startswith("%"):
                v = _lookup(toks[pos][1:], scopes)
                pos += 1
            elif toks[pos].lower() in _TEXT_OPS:
                v, pos = _parse_expr(toks, pos, scopes)
            else:
                lit = _literal(toks[pos])
                if lit is None:
                    raise SyntaxError(f"bad token {toks[pos]!r}")
                v = Lit(lit)
                pos += 1
            items.append(v)
        if pos >= len(toks):
            raise SyntaxError("unclosed list")
        return items, pos + 1
    if tok.startswith("%"):
        return _lookup(tok[1:], scopes), pos + 1
    lit = _literal(tok)
    if lit is None:
        raise SyntaxError(f"bad token {tok!r}")
    return Lit(lit), pos + 1



def _parse_expr(toks, pos, scopes):
    if pos >= len(toks):
        raise SyntaxError("unexpected end of expression")
    word = toks[pos].lower()
    if word not in _TEXT_OPS:
        return _parse_operand(toks, pos, scopes)

    op = word
    pos += 1
    args = []
    attrs = {}
    while pos < len(toks):
        tok = toks[pos]
        if tok in (")", "]"):
            break
        if tok == ":":
            pos += 1
            continue
        if "=" in tok and not tok.startswith("%"):
            k, v = tok.split("=", 1)
            if k in _STRING_ATTRS:
                attrs[k] = v
            else:
                lit = _literal(v)
                if lit is not None:
                    attrs[k] = lit
                else:
                    try:
                        attrs[k] = int(v, 0)
                    except ValueError:
                        attrs[k] = v
            pos += 1
            continue
        if tok in NP_DTYPE and "dtype" not in attrs:
            attrs["dtype"] = tok
            pos += 1
            continue
        if tok in NP_DTYPE and op == "cast":
            attrs["dtype"] = tok
            pos += 1
            continue
        if tok in ("(", "["):
            v, pos = _parse_operand(toks, pos, scopes)
            args.append(v)
            continue
        if tok.startswith("%"):
            args.append(_lookup(tok[1:], scopes))
            pos += 1
            continue
        lit = _literal(tok)
        if lit is not None:
            args.append(Lit(lit))
            pos += 1
            continue
        if tok.lower() in _TEXT_OPS:
            v, pos = _parse_expr(toks, pos, scopes)
            args.append(v)
            continue
        raise SyntaxError(f"bad token {tok!r} in {op}")
    return _build(op, args, attrs), pos


def parse_uir(text):
    lines = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)

    scopes = [{}]
    block_stack = []
    last_value = None
    result = None

    for line in lines:
        if line == "}":
            if not block_stack:
                raise SyntaxError("unmatched '}'")
            name, scope, prev_last = block_stack.pop()
            body = scope.get("result", prev_last)
            if body is None:
                raise SyntaxError("empty block")
            scopes.pop()
            if name is not None:
                scopes[-1][name] = body
            last_value = body
            continue

        m = re.match(r"%([A-Za-z_]\w*)\s*=\s*(.*)$", line)
        if m:
            name, rhs = m.group(1), m.group(2)
            if rhs.strip() == "{":
                block_stack.append((name, {}, None))
                scopes.append(scopes[-1].copy() if False else {})
                continue
            toks = _TOKEN.findall(rhs)
            val, pos = _parse_expr(toks, 0, scopes)
            if pos != len(toks):
                raise SyntaxError(f"trailing tokens in {line!r}")
            scopes[-1][name] = val
            last_value = val
            if block_stack:
                block_stack[-1] = (block_stack[-1][0], block_stack[-1][1], val)
            if name == "result":
                result = val
            continue

        raise SyntaxError(f"bad statement {line!r}")

    if block_stack:
        raise SyntaxError("unclosed block")

    if result is not None:
        return result
    if last_value is None:
        raise SyntaxError("empty program")
    return last_value
