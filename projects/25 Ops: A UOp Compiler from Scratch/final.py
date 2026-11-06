"""
25 Ops: A UOp Compiler from Scratch — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  UOp ──
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


sys.setrecursionlimit(100000)

ParamArg = namedtuple("ParamArg", ["slot", "dtype", "lo", "hi"])
BufArg = namedtuple("BufArg", ["slot", "dtype", "shape", "url", "sha256"])
I32_MIN, I32_MAX = -(1 << 31), (1 << 31) - 1


def wrap32(v):
    v = int(v) & 0xFFFFFFFF
    return v - (1 << 32) if v >= (1 << 31) else v


def _akey(a):
    if isinstance(a, tuple):
        return (type(a).__name__,) + tuple(_akey(x) for x in a)
    if isinstance(a, float):
        return ("float", "nan" if a != a else a.hex())
    return (type(a).__name__, a)


class UOp:
    _cache = {}
    __slots__ = ("op", "src", "arg", "_memo")

    def __new__(cls, op, src=(), arg=None):
        src = tuple(src)
        key = (op, src, _akey(arg))
        cached = UOp._cache.get(key)
        if cached is not None:
            return cached

        node = object.__new__(cls)
        node.op = op
        node.src = src
        node.arg = arg
        node._memo = {}
        UOp._cache[key] = node
        return node

    def __repr__(self):
        return f"UOp({self.op}, arg={self.arg!r}, src={len(self.src)})"

    def toposort(self):
        order = []
        seen = set()
        stack = [(self, False)]

        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue

            if id(node) in seen:
                continue

            seen.add(id(node))
            stack.append((node, True))
            for s in reversed(node.src):
                if id(s) not in seen:
                    stack.append((s, False))

        return order


def const(value, dtype=None):
    if dtype is None:
        if isinstance(value, (bool, np.bool_)):
            dtype = "bool"
        elif isinstance(value, (float, np.floating)):
            dtype = "f32"
        else:
            dtype = "i32"
    if dtype == "bool":
        if isinstance(value, (float, np.floating)):
            raise TypeError("a float cannot be a bool constant")
        v = bool(value)
    elif dtype == "i32":
        if isinstance(value, (bool, np.bool_, float, np.floating)):
            raise TypeError(f"{value!r} cannot be an i32 constant")
        v = wrap32(value)
    elif dtype == "f32":
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("a bool cannot be an f32 constant")
        v = float(np.float32(value))
    else:
        raise TypeError(f"unknown dtype {dtype!r}")
    return UOp("CONST", (), v)

# ── Step 002  infer ──
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


NP_DTYPE = {"bool": np.bool_, "i32": np.int32, "f32": np.float32}
ALU_OPS = ("ADD", "MUL", "MAX", "XOR", "CMPLT", "SHL", "DIV", "MOD", "WHERE", "CAST", "RECIP", "TRUNC")
TYPE_RULES = {}


def infer(u):
    memo = u._memo.get("type")
    if memo is not None:
        return memo

    rule = TYPE_RULES.get(u.op)
    if rule is None:
        raise TypeError(f"no type rule for {u.op}")

    result = rule(u)
    u._memo["type"] = result
    return result

UOp.dtype = property(lambda self: infer(self)[0])
UOp.shape = property(lambda self: infer(self)[1])


def _same(srcs, op):
    d, s = srcs[0].dtype, srcs[0].shape
    for x in srcs[1:]:
        if x.dtype != d or x.shape != s:
            raise TypeError(f"{op}: operand {x.dtype}{list(x.shape)} does not match {d}{list(s)}")
    return d, s


def _alu_rule(allowed, out=None):
    def rule(u):
        if len(u.src) != 2:
            raise TypeError(f"{u.op} takes two operands")
        d, s = _same(u.src, u.op)
        if d not in allowed:
            raise TypeError(f"{u.op} does not accept {d}")
        return (out or d), s
    return rule


def _unary_rule(allowed):
    def rule(u):
        if len(u.src) != 1 or u.src[0].dtype not in allowed:
            raise TypeError(f"{u.op} needs one {'/'.join(allowed)} operand")
        return u.src[0].dtype, u.src[0].shape
    return rule


def _scalar_i32(x, what):
    if x.dtype != "i32" or x.shape != ():
        raise TypeError(f"{what} must be an i32 scalar")


def _body_params(body):
    found, seen, stack = {}, set(), [body]
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        if u.op == "PARAM":
            if u.arg.slot in found:
                raise TypeError(f"two PARAMs with slot {u.arg.slot} in one function body")
            found[u.arg.slot] = u
        stack.extend(u.src[1:] if u.op == "CALL" else u.src)
    return found


def _call_rule(u):
    body, args = u.src[0], u.src[1:]
    for slot, p in _body_params(body).items():
        if slot >= len(args):
            raise TypeError(f"CALL is missing argument {slot}")
        if args[slot].dtype != p.arg.dtype or args[slot].shape != ():
            raise TypeError(f"CALL argument {slot} must be a {p.arg.dtype} scalar")
    return body.dtype, body.shape


def _stack_rule(u):
    if len(u.src) == 0:
        raise TypeError("STACK needs at least one source")

    d, s = _same(u.src, "STACK")
    return d, (len(u.src),) + s


def _index_rule(u):
    if len(u.src) < 2:
        raise TypeError("INDEX needs a source and at least one index")

    x = u.src[0]
    indices = u.src[1:]
    for i in indices:
        _scalar_i32(i, "INDEX index")

    k = len(indices)
    rank = len(x.shape)
    if k < 1 or k > rank:
        raise TypeError(f"INDEX pops {k} axes from a rank-{rank} value")

    return x.dtype, x.shape[k:]

def _shrink_rule(u):
    if len(u.src) < 2:
        raise TypeError("SHRINK needs a source and at least one begin")

    x = u.src[0]
    begins = u.src[1:]
    sizes = u.arg
    if not isinstance(sizes, tuple) or len(sizes) < 1:
        raise TypeError("SHRINK sizes must be a non-empty tuple")

    k = len(sizes)
    if len(begins) != k:
        raise TypeError(f"SHRINK got {len(begins)} begins for {k} sizes")

    if k > len(x.shape):
        raise TypeError("SHRINK window exceeds the source rank")

    for b in begins:
        _scalar_i32(b, "SHRINK begin")

    for n, ax in zip(sizes, x.shape[:k]):
        if not (0 < n <= ax):
            raise TypeError(f"SHRINK size {n} is not in 1..{ax}")

    return x.dtype, tuple(sizes) + x.shape[k:]

def _where_rule(u):
    c, a, b = u.src
    if c.dtype != "bool":
        raise TypeError("WHERE condition must be bool")
    d, s = _same((a, b), "WHERE")
    if c.shape != s:
        raise TypeError("WHERE condition shape must match its values")
    return d, s


def _cast_rule(u):
    if u.arg not in NP_DTYPE:
        raise TypeError(f"cannot CAST to {u.arg!r}")
    return u.arg, u.src[0].shape


TYPE_RULES.update({
    "CONST": lambda u: ("bool" if isinstance(u.arg, bool) else "f32" if isinstance(u.arg, float) else "i32", ()),
    "PARAM": lambda u: (u.arg.dtype, ()),
    "CALL": _call_rule,
    "STACK": _stack_rule,
    "INDEX": _index_rule,
    "SHRINK": _shrink_rule,
    "ADD": _alu_rule(("i32", "f32")),
    "MUL": _alu_rule(("i32", "f32")),
    "MAX": _alu_rule(("bool", "i32", "f32")),
    "XOR": _alu_rule(("bool", "i32")),
    "CMPLT": _alu_rule(("bool", "i32", "f32"), out="bool"),
    "SHL": _alu_rule(("i32",)),
    "DIV": _alu_rule(("i32",)),
    "MOD": _alu_rule(("i32",)),
    "WHERE": _where_rule,
    "CAST": _cast_rule,
    "RECIP": _unary_rule(("f32",)),
    "TRUNC": _unary_rule(("f32",)),
})

# ── Step 003  parse_uir ──
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


def _lit_tok(tok):
    """_literal + hex literallerini i32'ye sar."""
    v = _literal(tok)
    if isinstance(v, int) and not isinstance(v, bool) and "0x" in tok.lower():
        v = (v + (1 << 31)) % (1 << 32) - (1 << 31)
    return v


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
            if toks[pos].lower() in _TEXT_OPS:
                v, pos = _parse_expr(toks, pos, scopes)
            else:
                v, pos = _parse_operand(toks, pos, scopes)
            items.append(v)
        if pos >= len(toks):
            raise SyntaxError("unclosed list")
        return items, pos + 1
    if tok in (")", "]", "{", "}"):
        raise SyntaxError(f"unexpected {tok!r}")
    if tok.startswith("%"):
        return _lookup(tok[1:], scopes), pos + 1
    lit = _lit_tok(tok)
    if lit is None:
        raise SyntaxError(f"bad token {tok!r}")
    return Lit(lit), pos + 1


def _parse_expr(toks, pos, scopes):
    if pos >= len(toks):
        raise SyntaxError("unexpected end of expression")
    op = toks[pos].lower()
    if op not in _TEXT_OPS:
        return _parse_operand(toks, pos, scopes)
    pos += 1
    args, attrs = [], {}
    after_colon = False
    while pos < len(toks):
        tok = toks[pos]
        if tok in (")", "]"):
            break
        if tok.startswith(":"):
            after_colon = True
            tok = tok[1:]
            if not tok:
                pos += 1
                continue
        if "=" in tok and not tok.startswith("%"):
            k, v = tok.split("=", 1)
            if not k or not v:
                raise SyntaxError(f"bad attribute {tok!r}")
            if k in _STRING_ATTRS:
                attrs[k] = v
            else:
                lit = _lit_tok(v)
                attrs[k] = lit if lit is not None else v
            pos += 1
            continue
        if after_colon and tok in NP_DTYPE:
            attrs["dtype"] = tok
            pos += 1
            continue
        if after_colon:
            raise SyntaxError(f"operand {tok!r} after ':'")
        if tok.lower() in _TEXT_OPS:
            v, pos = _parse_expr(toks, pos, scopes)
        else:
            v, pos = _parse_operand(toks, pos, scopes)
        args.append(v)
    return _build(op, args, attrs), pos


def parse_uir(text):
    scopes = [{}]
    blocks = []
    last = None

    def bind(name, val):
        nonlocal last
        scopes[-1][name] = val
        last = val

    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line == "}":
            if not blocks:
                raise SyntaxError("unmatched '}'")
            name, outer_last = blocks.pop()
            body = scopes.pop().get("result", last)
            if body is None:
                raise SyntaxError("empty block")
            last = outer_last
            bind(name, _to_uop(body))
            continue
        m = re.match(r"%(\w+)\s*=\s*(.*)$", line)
        if not m:
            raise SyntaxError(f"bad statement {line!r}")
        name, rhs = m.group(1), m.group(2).strip()
        if rhs == "{":
            blocks.append((name, last))
            scopes.append({})
            last = None
            continue
        toks = _TOKEN.findall(rhs)
        if not toks:
            raise SyntaxError(f"empty expression in {line!r}")
        val, pos = _parse_expr(toks, 0, scopes)
        if pos != len(toks):
            raise SyntaxError(f"trailing tokens in {line!r}")
        bind(name, val)

    if blocks:
        raise SyntaxError("unclosed block")
    val = scopes[0].get("result", last)
    if val is None:
        raise SyntaxError("empty program")
    return _to_uop(val)

# ── Step 004  print_uir ──
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


def _lit_text(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _list_text(xs):
    return "[" + " ".join(str(x) for x in xs) + "]"


def print_uir(root):
    if root.op == "CONST":
        return f"%0 = const {_lit_text(root.arg)}\n"
    names = {}
    lines = []

    def ref(s):
        return _lit_text(s.arg) if s.op == "CONST" else names[s]

    for u in root.toposort():
        if u.op == "CONST":
            continue
        op = u.op
        srcs = " ".join(ref(s) for s in u.src)
        if op == "PARAM":
            slot, dtype, lo, hi = u.arg
            body = f"param [] : dtype={dtype} slot={slot}"
            if lo is not None:
                body += f" min={_lit_text(lo)}"
            if hi is not None:
                body += f" max={_lit_text(hi)}"
        elif op == "CAST":
            body = f"cast {srcs} : dtype={u.arg}"
        elif op in ("SHRINK", "PAD", "EXPAND", "RESHAPE", "PERMUTE"):
            body = f"{op.lower()} {srcs} {_list_text(u.arg)}"
        elif op in ("BUFFER", "ALLOC"):
            slot, dtype, shape, url, sha256 = u.arg
            body = f"{op.lower()} {_list_text(shape)} : dtype={dtype} slot={slot}"
            if url is not None:
                body += f" url={url}"
            if sha256 is not None:
                body += f" sha256={sha256}"
        elif op == "RANGE":
            body = f"range {srcs} : slot={u.arg}"
        elif op == "REDUCE":
            rop, pop = u.arg
            body = f"reduce {srcs} : op={str(rop).lower()} pop={pop}"
        else:
            body = f"{op.lower()} {srcs}" if srcs else op.lower()
        names[u] = f"%{len(lines)}"
        lines.append(f"{names[u]} = {body}")
    return "\n".join(lines) + "\n"

# ── Step 005  bounds ──
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


BOUNDS_RULES = {}


def _full(d):
    return (0, 1) if d == "bool" else (I32_MIN, I32_MAX) if d == "i32" else None


def _fit(lo, hi, d):
    return (lo, hi) if I32_MIN <= lo and hi <= I32_MAX else _full(d)


def bounds(u):
    if "bounds" in u._memo:
        return u._memo["bounds"]
    if u.dtype not in ("bool", "i32"):
        r = None
    else:
        rule = BOUNDS_RULES.get(u.op)
        r = rule(u) if rule is not None else _full(u.dtype)
    u._memo["bounds"] = r
    return r


def _corners(f, a, b, d):
    vals = [f(x, y) for x in a for y in b]
    return _fit(min(vals), max(vals), d)


def _b_div(u):
    a, b = bounds(u.src[0]), bounds(u.src[1])
    if b[0] > 0 or b[1] < 0:
        return _corners(lambda x, y: x // y, a, b, "i32")
    return _full("i32")


def _b_mod(u):
    a, b = bounds(u.src[0]), bounds(u.src[1])
    if b[0] > 0:
        c = b[0]
        if b[0] == b[1] and a[0] // c == a[1] // c:
            return (a[0] % c, a[1] % c)
        return (0, b[1] - 1)
    return _full("i32")


def _b_xor(u):
    a, b = bounds(u.src[0]), bounds(u.src[1])
    if u.dtype == "bool":
        return (a[0] ^ b[0],) * 2 if a[0] == a[1] and b[0] == b[1] else (0, 1)
    if a[0] >= 0 and b[0] >= 0:
        return (0, (1 << max(a[1], b[1]).bit_length()) - 1)
    return _full("i32")


def _b_shl(u):
    a, s = bounds(u.src[0]), bounds(u.src[1])
    if s[0] == s[1] and 0 <= s[0] <= 31:
        return _fit(a[0] << s[0], a[1] << s[0], "i32")
    return _full("i32")


def _b_cmplt(u):
    a, b = bounds(u.src[0]), bounds(u.src[1])
    if a is None or b is None:
        return (0, 1)
    if a[1] < b[0]:
        return (1, 1)
    if a[0] >= b[1]:
        return (0, 0)
    return (0, 1)


def _b_where(u):
    c = bounds(u.src[0])
    if c == (1, 1):
        return bounds(u.src[1])
    if c == (0, 0):
        return bounds(u.src[2])
    a, b = bounds(u.src[1]), bounds(u.src[2])
    return (min(a[0], b[0]), max(a[1], b[1]))


def _b_cast(u):
    x = u.src[0]
    if x.dtype == "f32":
        return _full(u.dtype)
    a = bounds(x)
    if u.dtype == "bool":
        return (0, 0) if a == (0, 0) else (1, 1) if a[0] > 0 or a[1] < 0 else (0, 1)
    return a


def _b_union(srcs):
    bs = [bounds(s) for s in srcs]
    return (min(b[0] for b in bs), max(b[1] for b in bs))


def _b_index(u):
    x, i = u.src[0], bounds(u.src[1])
    if x.op == "STACK" and len(u.src) == 2:
        picks = x.src[max(i[0], 0):min(i[1], len(x.src) - 1) + 1]
        return _b_union(picks) if picks else bounds(x)
    return bounds(x)


BOUNDS_RULES.update({
    "CONST": lambda u: (int(u.arg), int(u.arg)),
    "PARAM": lambda u: (_full(u.dtype)[0] if u.arg.lo is None else u.arg.lo, _full(u.dtype)[1] if u.arg.hi is None else u.arg.hi),
    "ADD": lambda u: _fit(bounds(u.src[0])[0] + bounds(u.src[1])[0], bounds(u.src[0])[1] + bounds(u.src[1])[1], "i32"),
    "MUL": lambda u: _corners(lambda x, y: x * y, bounds(u.src[0]), bounds(u.src[1]), "i32"),
    "DIV": _b_div,
    "MOD": _b_mod,
    "MAX": lambda u: (max(bounds(u.src[0])[0], bounds(u.src[1])[0]), max(bounds(u.src[0])[1], bounds(u.src[1])[1])),
    "XOR": _b_xor,
    "SHL": _b_shl,
    "CMPLT": _b_cmplt,
    "WHERE": _b_where,
    "CAST": _b_cast,
    "STACK": lambda u: _b_union(u.src),
    "INDEX": _b_index,
    "SHRINK": lambda u: bounds(u.src[0]),
    "CALL": lambda u: bounds(u.src[0]),
})

# ── Step 006  eval_uop ──
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


EVAL = {}


def _wrap(a):
    a = np.asarray(a, np.int64)
    return ((a + (1 << 31)) % (1 << 32) - (1 << 31)).astype(np.int32)


def _ieee_max(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    r = np.where(a > b, a, b)
    r = np.where((a == b) & np.signbit(b), a, r)
    r = np.where(np.isnan(a) | np.isnan(b), np.float32(np.nan), r)
    return np.asarray(r, np.float32)


def _ev_int(f):
    def ev(u, a, b):
        if u.op in ("DIV", "MOD") and np.any(b == 0):
            raise ZeroDivisionError(f"{u.op} by zero")
        if u.op == "SHL" and np.any((b < 0) | (b > 31)):
            raise ValueError("SHL amount outside 0..31")
        return _wrap(f(a.astype(np.int64), b.astype(np.int64)))
    return ev


def _ev_arith(f):
    def ev(u, a, b):
        if u.dtype == "i32":
            return _wrap(f(a.astype(np.int64), b.astype(np.int64)))
        return f(a, b).astype(np.float32)
    return ev


def _ev_cast(u, x):
    if u.arg == "bool":
        return x != 0
    if u.arg == "f32":
        return x.astype(np.float32)
    if u.src[0].dtype == "f32":
        t = np.trunc(x.astype(np.float64))
        if not np.all(np.isfinite(t)) or np.any((t < I32_MIN) | (t > I32_MAX)):
            raise ValueError("f32 -> i32 cast out of range")
        return t.astype(np.int64).astype(np.int32)
    return x.astype(np.int32)


def _check_index(i, n, what):
    i = int(i)
    if not 0 <= i < n:
        raise IndexError(f"{what} {i} is outside an axis of size {n}")
    return i


def _ev_index(u, x, *idx):
    return x[tuple(_check_index(i, x.shape[k], "INDEX") for k, i in enumerate(idx))]


def _ev_shrink(u, x, *begins):
    sl = []
    for k, (b, n) in enumerate(zip(begins, u.arg)):
        b = int(b)
        if b < 0 or b + n > x.shape[k]:
            raise IndexError(f"SHRINK window [{b}, {b + n}) is outside an axis of size {x.shape[k]}")
        sl.append(slice(b, b + n))
    return x[tuple(sl)]


EVAL.update({
    "CONST": lambda u: np.asarray(u.arg, NP_DTYPE[u.dtype]),
    "ADD": _ev_arith(lambda a, b: a + b),
    "MUL": _ev_arith(lambda a, b: a * b),
    "MAX": lambda u, a, b: _ieee_max(a, b) if u.dtype == "f32" else np.maximum(a, b),
    "XOR": lambda u, a, b: np.logical_xor(a, b) if u.dtype == "bool" else np.bitwise_xor(a, b),
    "CMPLT": lambda u, a, b: np.less(a, b),
    "SHL": _ev_int(np.left_shift),
    "DIV": _ev_int(np.floor_divide),
    "MOD": _ev_int(np.mod),
    "WHERE": lambda u, c, a, b: np.where(c, a, b).astype(NP_DTYPE[u.dtype]),
    "CAST": _ev_cast,
    "RECIP": lambda u, x: (np.float32(1.0) / x).astype(np.float32),
    "TRUNC": lambda u, x: np.trunc(x).astype(np.float32),
    "STACK": lambda u, *xs: np.stack(xs),
    "INDEX": _ev_index,
    "SHRINK": _ev_shrink,
})


def _param_value(p, v):
    a = p.arg
    if a.dtype == "i32":
        v = wrap32(int(v))
    arr = np.asarray(v, NP_DTYPE[a.dtype])
    if a.dtype == "i32" and ((a.lo is not None and int(arr) < a.lo) or (a.hi is not None and int(arr) > a.hi)):
        raise ValueError(f"argument {int(arr)} is outside PARAM slot {a.slot} bounds [{a.lo}, {a.hi}]")
    return arr


def eval_uop(root, inputs=None):
    inputs = {} if inputs is None else inputs

    def ev(u, env, memo):
        key = id(u)
        if key in memo:
            return memo[key]
        if u.op == "PARAM":
            slot = u.arg.slot
            if slot not in env:
                raise KeyError(slot)
            r = _param_value(u, env[slot])
        elif u.op == "CALL":
            args = [ev(s, env, memo) for s in u.src[1:]]
            r = ev(u.src[0], dict(enumerate(args)), {})
        else:
            fn = EVAL.get(u.op)
            if fn is None:
                raise NotImplementedError(f"no EVAL rule for {u.op}")
            vals = [ev(s, env, memo) for s in u.src]
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                r = np.asarray(fn(u, *vals))
        memo[key] = r
        return r

    return ev(root, dict(inputs), {})

# ── Step 007  graph_rewrite ──
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


def graph_rewrite(root, rules, max_steps=10000):
    memo = {}

    def rebuild(u):
        new_src = tuple(rw(s) for s in u.src)
        if all(a is b for a, b in zip(new_src, u.src)):
            return u
        return UOp(u.op, new_src, u.arg)

    def rw(u):
        key = id(u)
        if key in memo:
            return memo[key][1]
        cur = rebuild(u)
        steps = 0
        while True:
            new = None
            for rule in rules:
                r = rule(cur)
                if r is not None and r is not cur:
                    new = r
                    break
            if new is None:
                break
            steps += 1
            if steps >= max_steps:
                raise RuntimeError("graph_rewrite did not reach a fixed point")
            if id(new) in memo:
                cur = memo[id(new)][1]
            else:
                cur = rebuild(new)
        memo[key] = (u, cur)
        return cur

    return rw(root)

# ── Step 008  fold_constants ──
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


def fold_constants(u):
    if u.op == "WHERE" and u.src[0].op == "CONST":
        return u.src[1] if u.src[0].arg else u.src[2]
    if u.op in ALU_OPS and u.src and all(s.op == "CONST" for s in u.src):
        try:
            value = eval_uop(u).item()
        except (ZeroDivisionError, ValueError, IndexError):
            return None
        return const(value, u.dtype)
    if u.op == "INDEX" and len(u.src) >= 2 and u.src[0].op == "STACK" and u.src[1].op == "CONST":
        x, k = u.src[0], u.src[1].arg
        if isinstance(k, bool) or not 0 <= k < len(x.src):
            return None
        elem = x.src[k]
        rest = u.src[2:]
        return UOp("INDEX", (elem,) + tuple(rest)) if rest else elem
    if u.op == "SHRINK" and len(u.src) == 2 and u.src[0].op == "STACK" and u.src[1].op == "CONST":
        x, b, n = u.src[0], u.src[1].arg, u.arg[0]
        if 0 <= b and b + n <= len(x.src):
            return UOp("STACK", tuple(x.src[b:b + n]))
    return None

# ── Step 009  symbolic ──
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


_COMMUTATIVE = ("ADD", "MUL", "MAX", "XOR")


def _is_const(u, v):
    return u.op == "CONST" and type(u.arg) is type(v) and u.arg == v


def _exact(x):
    """True when an i32 ADD/MUL tree is computed without wrapping around."""
    if x.op in ("ADD", "MUL"):
        a, b = bounds(x.src[0]), bounds(x.src[1])
        vals = [a[0] + b[0], a[1] + b[1]] if x.op == "ADD" else [p * q for p in a for q in b]
        return _exact(x.src[0]) and _exact(x.src[1]) and I32_MIN <= min(vals) and max(vals) <= I32_MAX
    return True


def _pos_int_const(u):
    return u.op == "CONST" and type(u.arg) is int and u.arg > 0


def _add(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return UOp("ADD", (a, b))


def _split_multiple(a, c):
    """Write a as q*c + r. Returns (q, r) with r None when a is an exact multiple of c, or None."""
    if a.op == "CONST" and type(a.arg) is int:
        if a.arg % c == 0:
            return const(a.arg // c), None
        return None
    if a.op == "MUL" and a.src[1].op == "CONST" and type(a.src[1].arg) is int:
        k = a.src[1].arg
        if k % c == 0:
            x = a.src[0]
            return (x, None) if k == c else (UOp("MUL", (x, const(k // c))), None)
        return None
    if a.op == "ADD":
        p, s = a.src
        sp, ss = _split_multiple(p, c), _split_multiple(s, c)
        if sp is None and ss is None:
            return None
        if sp is None:
            return ss[0], _add(p, ss[1])
        if ss is None:
            return sp[0], _add(sp[1], s)
        return _add(sp[0], ss[0]), _add(sp[1], ss[1])
    return None


def symbolic(u):
    if u.op not in ALU_OPS or u.op == "CONST":
        return None
    src = u.src
    scalar = u.shape == ()

    if scalar and u.dtype in ("i32", "bool"):
        b = bounds(u)
        if b is not None and b[0] == b[1]:
            return const(b[0], u.dtype)

    if u.op == "WHERE" and src[1] is src[2]:
        return src[1]

    if u.op in _COMMUTATIVE and len(src) == 2:
        if src[0].op == "CONST" and src[1].op != "CONST":
            return UOp(u.op, (src[1], src[0]))
        if u.op == "MAX" and src[0] is src[1]:
            return src[0]

    if u.dtype == "i32" and len(src) == 2:
        x, y = src
        if u.op == "ADD" and _is_const(y, 0):
            return x
        if u.op == "MUL" and _is_const(y, 1):
            return x
        if u.op == "DIV" and _is_const(y, 1):
            return x
        if u.op == "XOR" and _is_const(y, 0):
            return x
        if u.op == "MUL" and _is_const(y, 0) and scalar:
            return const(0)
        if u.op == "MOD" and _is_const(y, 1) and scalar:
            return const(0)
        if u.op == "XOR" and x is y and scalar:
            return const(0)

    if u.op == "MAX" and u.dtype == "i32" and scalar:
        a, b = bounds(src[0]), bounds(src[1])
        if a[0] >= b[1]:
            return src[0]
        if b[0] >= a[1]:
            return src[1]

    if u.op in ("ADD", "MUL") and u.dtype == "i32":
        x, y = src
        if y.op == "CONST" and x.op == u.op and x.src[1].op == "CONST" and x.src[0].op != "CONST":
            return UOp(u.op, (x.src[0], UOp(u.op, (x.src[1], y))))
        if u.op == "ADD":
            if x.op == "ADD" and x.src[1].op == "CONST" and x.src[0].op != "CONST" and y.op != "CONST":
                return UOp("ADD", (UOp("ADD", (x.src[0], y)), x.src[1]))
            if y.op == "ADD" and y.src[1].op == "CONST" and y.src[0].op != "CONST" and x.op != "CONST":
                return UOp("ADD", (UOp("ADD", (x, y.src[0])), y.src[1]))

    if u.op in ("DIV", "MOD") and u.dtype == "i32" and scalar and _pos_int_const(src[1]):
        x, c = src[0], src[1].arg
        if u.op == "MOD":
            bx = bounds(x)
            if bx[0] >= 0 and bx[1] < c:
                return x
        if _exact(x):
            sp = _split_multiple(x, c)
            if sp is not None:
                q, r = sp
                if u.op == "DIV":
                    return q if r is None else UOp("ADD", (q, UOp("DIV", (r, src[1]))))
                return const(0) if r is None else UOp("MOD", (r, src[1]))
        if u.op == "DIV" and x.op == "DIV" and _pos_int_const(x.src[1]):
            ac = x.src[1].arg * c
            if ac <= I32_MAX:
                return UOp("DIV", (x.src[0], const(ac)))

    if u.dtype == "f32" and len(src) == 2:
        x, y = src
        if u.op == "MUL" and _is_const(y, 1.0):
            return x
        if u.op == "ADD" and y.op == "CONST" and type(y.arg) is float and y.arg == 0.0 and math.copysign(1.0, y.arg) < 0:
            return x

    return None

# ── Step 010  inline_calls ──
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


def substitute(root, mapping):
    repl = {id(k): v for k, v in mapping.items()}
    keep = list(mapping.keys())
    memo = {}

    def sub(u):
        key = id(u)
        if key in memo:
            return memo[key][1]
        if key in repl:
            r = repl[key]
        else:
            if u.op == "CALL":
                new_src = (u.src[0],) + tuple(sub(s) for s in u.src[1:])
            else:
                new_src = tuple(sub(s) for s in u.src)
            if all(a is b for a, b in zip(new_src, u.src)):
                r = u
            else:
                r = UOp(u.op, new_src, u.arg)
        memo[key] = (u, r)
        return r

    out = sub(root)
    del keep
    return out


def inline_calls(u):
    if u.op != "CALL":
        return None
    body, args = u.src[0], u.src[1:]
    params = _body_params(body)
    mapping = {}
    for slot, p in params.items():
        if slot >= len(args):
            raise TypeError(f"CALL is missing the argument for PARAM slot {slot}")
        mapping[p] = args[slot]
    return substitute(body, mapping)


def run_by_rewriting(root, inputs=None):
    inputs = {} if inputs is None else inputs
    mapping = {}
    for slot, p in _body_params(root).items():
        if slot not in inputs:
            raise KeyError(slot)
        mapping[p] = const(_param_value(p, inputs[slot]).item(), p.dtype)
    return graph_rewrite(substitute(root, mapping), [inline_calls, fold_constants, symbolic])

# ── Step 011  render_c ──
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


C_TYPES = {"bool": "_Bool", "i32": "int32_t", "f32": "float"}
C_HEADER = """#include <stdint.h>
#include <math.h>
static inline int32_t idiv32(int32_t a, int32_t b) { if (b == -1) return (int32_t)(0u - (uint32_t)a); int32_t q = a / b; if ((a % b != 0) && ((a < 0) != (b < 0))) q -= 1; return q; }
static inline int32_t imod32(int32_t a, int32_t b) { if (b == -1) return 0; int32_t r = a % b; if (r != 0 && ((r < 0) != (b < 0))) r += b; return r; }
static inline float fmax32(float a, float b) { if (a != a || b != b) return NAN; if (a == b) return signbit(a) ? b : a; return a > b ? a : b; }
"""


def c_literal(u):
    v = u.arg
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        if v != v:
            return "NAN"
        if v in (float("inf"), float("-inf")):
            return "INFINITY" if v > 0 else "(-INFINITY)"
        return float.hex(v) + "f"
    return "(-2147483647-1)" if v == I32_MIN else str(v)


def c_expr(u, a):
    op = u.op
    if op in ("ADD", "MUL"):
        sym = "+" if op == "ADD" else "*"
        if u.dtype == "i32":
            return f"(int32_t)((uint32_t)({a[0]}) {sym} (uint32_t)({a[1]}))"
        return f"({a[0]} {sym} {a[1]})"
    if op == "SHL":
        return f"(int32_t)((uint32_t)({a[0]}) << ({a[1]}))"
    if op == "DIV":
        return f"idiv32({a[0]}, {a[1]})"
    if op == "MOD":
        return f"imod32({a[0]}, {a[1]})"
    if op == "MAX":
        if u.dtype == "f32":
            return f"fmax32({a[0]}, {a[1]})"
        return f"(({a[0]}) > ({a[1]}) ? ({a[0]}) : ({a[1]}))"
    if op == "XOR":
        return f"(({a[0]}) ^ ({a[1]}))"
    if op == "CMPLT":
        return f"(({a[0]}) < ({a[1]}))"
    if op == "WHERE":
        return f"(({a[0]}) ? ({a[1]}) : ({a[2]}))"
    if op == "CAST":
        if u.arg == "f32":
            return f"(float)({a[0]})"
        if u.arg == "i32":
            return f"(int32_t)({a[0]})"
        return f"(({a[0]}) != 0)"
    if op == "RECIP":
        return f"(1.0f / ({a[0]}))"
    if op == "TRUNC":
        return f"truncf({a[0]})"
    raise NotImplementedError(f"no C expression for {op}")


def render_c(root, name="program"):
    root = graph_rewrite(root, [inline_calls])
    names = {}
    lines = []
    k = 0
    for u in root.toposort():
        if u.op == "CONST":
            names[u] = c_literal(u)
            continue
        if u.op == "PARAM":
            names[u] = f"p{u.arg.slot}"
            continue
        t = f"t{k}"
        k += 1
        T = C_TYPES[u.dtype]
        a = [names[s] for s in u.src]
        if u.op == "STACK":
            if len(u.shape) != 1:
                raise NotImplementedError("only rank-1 STACKs can be rendered")
            prefix = "static const " if all(s.op == "CONST" for s in u.src) else ""
            lines.append(f"{prefix}{T} {t}[{len(u.src)}] = {{{', '.join(a)}}};")
        elif u.op == "INDEX":
            lines.append(f"{T} {t} = {a[0]}[{a[1]}];")
        elif u.op == "SHRINK":
            lines.append(f"const {T}* {t} = {a[0]} + {a[1]};")
        elif u.op in ALU_OPS:
            lines.append(f"{T} {t} = {c_expr(u, a)};")
        else:
            raise NotImplementedError(f"cannot render {u.op} in C")
        names[u] = t

    params = [p for _, p in sorted(_body_params(root).items())]
    args = [f"{C_TYPES[p.dtype]} p{p.arg.slot}" for p in params]
    T = C_TYPES[root.dtype]
    if root.shape == ():
        head = f"{T} {name}({', '.join(args) if args else 'void'})"
        lines.append(f"return {names[root]};")
    elif len(root.shape) == 1:
        head = f"void {name}({', '.join([f'{T}* out'] + args)})"
        lines.append(f"for (int k = 0; k < {root.shape[0]}; k++) out[k] = {names[root]}[k];")
    else:
        raise NotImplementedError("only scalar and rank-1 roots can be rendered")
    return C_HEADER + head + " {\n" + "\n".join("  " + l for l in lines) + "\n}\n"

# ── Step 012  compile_c ──
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


_LIBS = {}
C_CTYPES = {"bool": ctypes.c_bool, "i32": ctypes.c_int32, "f32": ctypes.c_float}


def find_cc():
    cc = os.environ.get("CC") or shutil.which("clang") or shutil.which("gcc") or shutil.which("cc")
    if not cc:
        raise RuntimeError("no C compiler found")
    return cc


def compile_c(src):
    key = hashlib.sha256(src.encode()).hexdigest()
    if key in _LIBS:
        return _LIBS[key]
    d = tempfile.mkdtemp()
    c_path = os.path.join(d, "prog.c")
    so_path = os.path.join(d, "prog.so")
    with open(c_path, "w") as f:
        f.write(src)
    proc = subprocess.run(
        [find_cc(), "-O2", "-ffp-contract=off", "-shared", "-fPIC", "-w", c_path, "-o", so_path, "-lm"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError("C compile failed:\n" + proc.stderr)
    lib = ctypes.CDLL(so_path)
    _LIBS[key] = lib
    return lib


def run_c(root, inputs=None, name="program"):
    inputs = {} if inputs is None else inputs
    lib = compile_c(render_c(root, name))
    fn = getattr(lib, name)
    params = [p for _, p in sorted(_body_params(root).items())]
    types = [C_CTYPES[p.dtype] for p in params]
    args = [C_CTYPES[p.dtype](_param_value(p, inputs[p.arg.slot]).item()) for p in params]
    if root.shape == ():
        fn.argtypes = types
        fn.restype = C_CTYPES[root.dtype]
        res = fn(*args)
        return np.asarray(res, NP_DTYPE[root.dtype])
    if len(root.shape) == 1:
        out = np.zeros(root.shape, NP_DTYPE[root.dtype])
        fn.argtypes = [ctypes.c_void_p] + types
        fn.restype = None
        fn(out.ctypes.data_as(ctypes.c_void_p), *args)
        return out
    raise NotImplementedError("only scalar and rank-1 roots can be run")

# ── Step 013  threefry_program ──
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


TF_ROTATIONS = (13, 15, 26, 6, 17, 29, 16, 24)


def threefry_block():
    """.uir text defining %threefry(k0, k1, c0, c1) -> [x0 x1] with ADD/MUL/DIV/MOD/INDEX/STACK/CALL only."""
    if sys.getrecursionlimit() < 20000:
        sys.setrecursionlimit(20000)
    lines = []
    counter = itertools.count()

    def fresh():
        return f"%v{next(counter)}"

    def emit(name, expr):
        lines.append(f"  {name} = {expr}")
        return name

    def nibble(a, i):
        return f"(mod {a} 16)" if i == 0 else f"(mod (div {a} {16 ** i}) 16)"

    def xor(a, b):
        terms = []
        for i in range(8):
            e = f"index %tx (add (mul {nibble(a, i)} 16) {nibble(b, i)})"
            if i == 0:
                terms.append(emit(fresh(), e))
            else:
                terms.append(emit(fresh(), f"mul ({e}) {16 ** i}"))
        while len(terms) > 1:
            nxt = []
            for j in range(0, len(terms), 2):
                if j + 1 < len(terms):
                    nxt.append(emit(fresh(), f"add {terms[j]} {terms[j + 1]}"))
                else:
                    nxt.append(terms[j])
            terms = nxt
        return terms[0]

    lines.append("%threefry = {")
    lines.append("  %k0 = param [] : dtype=i32 slot=0")
    lines.append("  %k1 = param [] : dtype=i32 slot=1")
    lines.append("  %c0 = param [] : dtype=i32 slot=2")
    lines.append("  %c1 = param [] : dtype=i32 slot=3")
    lines.append("  %tx = [" + " ".join(str(p ^ q) for p in range(16) for q in range(16)) + "]")

    k01 = xor("%k0", "%k1")
    ks2 = xor(k01, str(0x1BD11BDA))
    ks = ["%k0", "%k1", ks2]

    emit("%x0", "add %c0 %k0")
    emit("%x1", "add %c1 %k1")
    for s in range(1, 6):
        rots = TF_ROTATIONS[:4] if s % 2 == 1 else TF_ROTATIONS[4:]
        for r in rots:
            emit("%x0", "add %x0 %x1")
            emit("%x1", f"add (mul %x1 {2 ** r}) (mod (div %x1 {2 ** (32 - r)}) {2 ** r})")
            emit("%x1", xor("%x1", "%x0"))
        emit("%x0", f"add %x0 {ks[s % 3]}")
        emit("%x1", f"add %x1 (add {ks[(s + 1) % 3]} {s})")
    lines.append("  %result = [%x0 %x1]")
    lines.append("}")
    return "\n".join(lines) + "\n"


def threefry_program():
    return threefry_block() + "\n".join([
        "%key0 = param [] : dtype=i32 slot=0",
        "%key1 = param [] : dtype=i32 slot=1",
        "%ctr0 = param [] : dtype=i32 slot=2",
        "%ctr1 = param [] : dtype=i32 slot=3",
        "%result = call %threefry %key0 %key1 %ctr0 %ctr1",
    ]) + "\n"

# ── Step 014  verify_memory ──
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


VIEW_OPS = ("AFTER", "INDEX", "SHRINK", "RESHAPE", "PERMUTE", "EXPAND", "PAD")


def addrspace(u):
    m = u._memo
    if "addrspace" not in m:
        if u.op in ("BUFFER", "ALLOC"):
            m["addrspace"] = "GLOBAL"
        elif u.op in VIEW_OPS and u.src and addrspace(u.src[0]) == "GLOBAL":
            m["addrspace"] = "GLOBAL"
        else:
            m["addrspace"] = "ALU"
    return m["addrspace"]


def base_of(view):
    while view.op in VIEW_OPS:
        view = view.src[0]
    if view.op not in ("BUFFER", "ALLOC"):
        raise TypeError(f"{view.op} is not a buffer view")
    return view


def _load_rule(u):
    if addrspace(u.src[0]) != "GLOBAL":
        raise TypeError("LOAD reads a BUFFER, an ALLOC or a view of one")
    return u.src[0].dtype, u.src[0].shape


def _store_rule(u):
    dst, val = u.src
    if addrspace(dst) != "GLOBAL":
        raise TypeError("STORE writes into a BUFFER, an ALLOC or a view of one")
    if (val.dtype, val.shape) != (dst.dtype, dst.shape):
        raise TypeError(f"STORE of {val.dtype}{list(val.shape)} into {dst.dtype}{list(dst.shape)}")
    return "void", ()


def _after_rule(u):
    for e in u.src[1:]:
        if e.dtype != "void" and e.op != "RANGE":
            raise TypeError("AFTER waits for effects (STOREs or ENDs) or names a loop (RANGE)")
    return u.src[0].dtype, u.src[0].shape


UOp.addrspace = property(addrspace)
TYPE_RULES.update({
    "BUFFER": lambda u: (u.arg.dtype, tuple(u.arg.shape)),
    "ALLOC": lambda u: (u.arg.dtype, tuple(u.arg.shape)),
    "LOAD": _load_rule,
    "STORE": _store_rule,
    "AFTER": _after_rule,
})


def verify_memory(root):
    errors = []
    nodes = root.toposort()

    for u in nodes:
        if u.op == "INDEX":
            shape = u.src[0].shape
            for k, i in enumerate(u.src[1:]):
                n = shape[k]
                b = bounds(i)
                if b is None or b[0] < 0 or b[1] > n - 1:
                    lo, hi = b if b is not None else ("?", "?")
                    errors.append(f"INDEX {lo}..{hi} may leave an axis of size {n}")
        elif u.op == "SHRINK":
            shape = u.src[0].shape
            for k, (bg, s) in enumerate(zip(u.src[1:], u.arg)):
                n = shape[k]
                b = bounds(bg)
                if b is None or b[0] < 0 or b[1] + s > n:
                    lo, hi = (b[0], b[1] + s) if b is not None else ("?", "?")
                    errors.append(f"SHRINK window {lo}..{hi} may leave an axis of size {n}")

    cones = {}

    def cone(u):
        key = id(u)
        if key not in cones:
            cones[key] = {id(x) for x in u.toposort()}
        return cones[key]

    groups = {}
    order = []
    for u in nodes:
        if u.op in ("LOAD", "STORE"):
            base = base_of(u.src[0])
            key = id(base)
            if key not in groups:
                groups[key] = (base, [])
                order.append(key)
            groups[key][1].append(u)

    for key in order:
        base, accesses = groups[key]
        for i in range(len(accesses)):
            for j in range(i + 1, len(accesses)):
                a, b = accesses[i], accesses[j]
                if a.op != "STORE" and b.op != "STORE":
                    continue
                if id(a) in cone(b) or id(b) in cone(a):
                    continue
                other = b if a.op == "STORE" else a
                errors.append(f"{other.op} and a STORE to buffer slot {base.arg.slot} are not ordered")
    return errors

# ── Step 015  emit_c ──
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


INLINE_OPS = ("CONST", "PARAM", "BUFFER", "RANGE")


def strides(shape):
    out, acc = [], 1
    for d in reversed(shape):
        out.append(acc)
        acc *= d
    return tuple(reversed(out))


def emit_c(u, names):
    op = u.op
    if op == "CONST":
        names[u] = c_literal(u)
        return None
    if op == "PARAM":
        names[u] = f"p{u.arg.slot}"
        return None
    if op == "BUFFER":
        names[u] = f"b{u.arg.slot}"
        return None
    if op == "RANGE":
        names[u] = f"r{u.arg}"
        return None
    if op == "ALLOC":
        name = f"a{u.arg.slot}"
        names[u] = name
        return f"static {C_TYPES[u.dtype]} {name}[{prod(u.arg.shape)}];"

    if addrspace(u) == "GLOBAL":
        src = u.src[0]
        if op in ("AFTER", "RESHAPE"):
            names[u] = names[src]
            return None
        if op in ("INDEX", "SHRINK"):
            st = strides(src.shape)
            terms = "".join(f" + ({names[i]}) * {st[k]}" for k, i in enumerate(u.src[1:]))
            names[u] = "(" + names[src] + terms + ")"
            return None
        raise NotImplementedError(f"{op} view is not a single pointer")

    T = C_TYPES.get(u.dtype)
    t = f"t{len(names)}"

    if op == "LOAD":
        names[u] = t
        return f"{T} {t} = *{names[u.src[0]]};"
    if op == "STORE":
        return f"*{names[u.src[0]]} = {names[u.src[1]]};"
    if op == "STACK":
        if len(u.shape) != 1:
            raise NotImplementedError("only rank-1 STACKs can be emitted")
        prefix = "static const " if all(s.op == "CONST" for s in u.src) else ""
        names[u] = t
        return f"{prefix}{T} {t}[{len(u.src)}] = {{{', '.join(names[s] for s in u.src)}}};"
    if op == "INDEX":
        names[u] = t
        return f"{T} {t} = {names[u.src[0]]}[{names[u.src[1]]}];"
    if op == "SHRINK":
        names[u] = t
        return f"const {T}* {t} = {names[u.src[0]]} + {names[u.src[1]]};"
    if op in ALU_OPS:
        names[u] = t
        return f"{T} {t} = {c_expr(u, [names[s] for s in u.src])};"
    raise NotImplementedError(f"cannot emit {op} in C")

# ── Step 016  render_kernel ──
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


def kernel_args(root):
    nodes = root.toposort()
    bufs = sorted({u.arg.slot: u for u in nodes if u.op == "BUFFER"}.values(), key=lambda u: u.arg.slot)
    params = sorted({u.arg.slot: u for u in nodes if u.op == "PARAM"}.values(), key=lambda u: u.arg.slot)
    return bufs, params


def _returns_value(root):
    return root.dtype != "void" and addrspace(root) != "GLOBAL"


def render_kernel(root, name="kernel", items=None):
    if items is None:
        items = [("op", u) for u in root.toposort() if u.op not in INLINE_OPS]
    bufs, params = kernel_args(root)
    names = {}
    for u in bufs + params:
        emit_c(u, names)
    lines = []

    def emit_items(its, depth):
        pad = "  " * depth
        for item in its:
            if item[0] == "loop":
                _, r, body = item
                if r not in names:
                    emit_c(r, names)
                rv = names[r]
                n = r.src[0].arg
                lines.append(f"{pad}for (int32_t {rv} = 0; {rv} < {n}; {rv}++) {{")
                emit_items(body, depth + 1)
                lines.append(f"{pad}}}")
                continue
            _, u = item
            if u.op == "END":
                raise ValueError("loops need a schedule: pass items=schedule(root)")
            for s in u.src:
                if s.op in INLINE_OPS and s not in names:
                    emit_c(s, names)
            stmt = emit_c(u, names)
            if stmt:
                lines.append(pad + stmt)

    emit_items(items, 1)

    args = [f"{C_TYPES[b.dtype]}* b{b.arg.slot}" for b in bufs]
    args += [f"{C_TYPES[p.dtype]} p{p.arg.slot}" for p in params]
    arglist = ", ".join(args) if args else "void"
    if _returns_value(root):
        head = f"{C_TYPES[root.dtype]} {name}({arglist})"
        lines.append(f"  return {names[root]};")
    else:
        head = f"void {name}({arglist})"
    return C_HEADER + head + " {\n" + "\n".join(lines) + "\n}\n"


def call_kernel(lib, name, root, buffers=None, params=None):
    buffers = {} if buffers is None else buffers
    params = {} if params is None else params
    bufs, ps = kernel_args(root)
    argtypes, args = [], []
    for b in bufs:
        slot = b.arg.slot
        dt = np.dtype(NP_DTYPE[b.dtype])
        n = prod(b.arg.shape)
        if slot not in buffers:
            buffers[slot] = np.zeros(tuple(b.arg.shape), dt)
        arr = buffers[slot]
        if not isinstance(arr, np.ndarray) or arr.dtype != dt or arr.size != n or not arr.flags["C_CONTIGUOUS"]:
            raise TypeError(f"buffer slot {slot} must be a C-contiguous {dt} array with {n} elements")
        argtypes.append(ctypes.c_void_p)
        args.append(arr.ctypes.data_as(ctypes.c_void_p))
    for p in ps:
        argtypes.append(C_CTYPES[p.dtype])
        args.append(C_CTYPES[p.dtype](_param_value(p, params[p.arg.slot]).item()))
    fn = getattr(lib, name)
    fn.argtypes = argtypes
    if _returns_value(root):
        fn.restype = C_CTYPES[root.dtype]
        res = fn(*args)
        return np.asarray(res, NP_DTYPE[root.dtype])[()]
    fn.restype = None
    fn(*args)
    return buffers

# ── Step 017  check_scopes ──
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


def _range_rule(u):
    n = u.src[0]
    if n.op != "CONST" or n.dtype != "i32" or n.arg < 1:
        raise TypeError("RANGE needs a positive integer constant end")
    return "i32", ()


def _end_rule(u):
    if len(u.src) != 2 or u.src[1].op != "RANGE":
        raise TypeError("END takes (value, RANGE)")
    return u.src[0].dtype, u.src[0].shape


TYPE_RULES.update({"RANGE": _range_rule, "END": _end_rule})
BOUNDS_RULES["RANGE"] = lambda u: (0, u.src[0].arg - 1)


def ranges_of(u):
    if "ranges" in u._memo:
        return u._memo["ranges"]
    for n in u.toposort():
        m = n._memo
        if "ranges" in m:
            continue
        if n.op == "RANGE":
            r = frozenset((n,))
        elif n.op == "END":
            r = n.src[0]._memo["ranges"] - {n.src[1]}
        else:
            srcs = n.src[1:] if n.op == "CALL" else n.src
            r = frozenset()
            for s in srcs:
                sr = s._memo["ranges"]
                if sr:
                    r = r | sr
        m["ranges"] = r
    return u._memo["ranges"]


def check_scopes(root):
    errors = []
    closes = {}
    order = []
    for u in root.toposort():
        if u.op != "END":
            continue
        if u.dtype != "void":
            errors.append("this compiler only closes loops of effects: END of a non-void value")
        r = u.src[1]
        if id(r) not in closes:
            closes[id(r)] = [r, 0]
            order.append(id(r))
        closes[id(r)][1] += 1
    for key in order:
        r, n = closes[key]
        if n > 1:
            errors.append(f"RANGE slot {r.arg} is closed by {n} ENDs")
    for r in sorted(ranges_of(root), key=lambda r: r.arg):
        errors.append(f"a value of RANGE slot {r.arg} is used outside its END")
    return errors

# ── Step 018  schedule ──
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


def schedule(root):
    if sys.getrecursionlimit() < 20000:
        sys.setrecursionlimit(20000)
    top = (None, [])
    placed = {}

    def target_of(u, scopes, srcs):
        rs = ranges_of(u)
        t = 0
        for i in range(len(scopes) - 1, 0, -1):
            if scopes[i][0] in rs:
                t = i
                break
        for s in srcs:
            if id(s) in placed:
                t = max(t, min(placed[id(s)][1], len(scopes) - 1))
        return t

    def walk(u, scopes):
        if id(u) in placed or u.op in INLINE_OPS:
            return
        if u.op == "END":
            v, r = u.src
            t = target_of(u, scopes, [s for s in u.src if id(s) in placed])
            body = []
            walk(v, scopes[:t + 1] + [(r, body)])
            scopes[t][1].append(("loop", r, body))
            placed[id(u)] = (u, t)
            return
        for s in u.src:
            walk(s, scopes)
        t = target_of(u, scopes, u.src)
        scopes[t][1].append(("op", u))
        placed[id(u)] = (u, t)

    walk(root, [top])
    return top[1]

# ── Step 019  run_program ──
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


def run_program(root, buffers=None, params=None, name="kernel"):
    errors = check_scopes(root) + verify_memory(root)
    if errors:
        raise ValueError("invalid program:\n  " + "\n  ".join(errors))
    lib = compile_c(render_kernel(root, name, schedule(root)))
    return call_kernel(lib, name, root, buffers, params)

# ── Step 020  gemm_program ──
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


def gemm_program(M, N, K):
    return "\n".join([
        f"%A = buffer {M * K} : dtype=f32 slot=0",
        f"%B = buffer {K * N} : dtype=f32 slot=1",
        f"%C = buffer {M * N} : dtype=f32 slot=2",
        f"%i = range {M} : slot=0",
        f"%j = range {N} : slot=1",
        f"%k = range {K} : slot=2",
        f"%at = add (mul %i {N}) %j",
        "%zero = store (index %C %at) 0.0",
        "%acc = index (after %C %zero %k) %at",
        f"%prod = mul (load (index %A (add (mul %i {K}) %k))) (load (index %B (add (mul %k {N}) %j)))",
        "%step = store %acc (add (load %acc) %prod)",
        "%result = after %C (end (end (end %step %k) %j) %i)",
    ]) + "\n"

# ── Step 021  lower_index ──
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


def _reduce_rule(u):
    x, (op, pop) = u.src[0], u.arg
    if op not in ("ADD", "MAX") or not 1 <= pop <= len(x.shape):
        raise TypeError("REDUCE needs op add|max and 1 <= pop <= rank")
    if op == "ADD" and x.dtype not in ("i32", "f32"):
        raise TypeError("REDUCE add needs i32 or f32")
    return x.dtype, x.shape[pop:]


def _expand_rule(u):
    x, shape = u.src[0], tuple(u.arg)
    if len(shape) != len(x.shape) or any(a != b and a != 1 for a, b in zip(x.shape, shape)):
        raise TypeError(f"cannot EXPAND {list(x.shape)} to {list(shape)}")
    return x.dtype, shape


def _identity(op, d):
    if op == "ADD":
        return 0.0 if d == "f32" else 0
    return float("-inf") if d == "f32" else (I32_MIN if d == "i32" else False)


def _ev_reduce(u, x):
    op, pop = u.arg
    rest, d = x.shape[pop:], u.dtype
    flat = x.reshape((-1,) + rest)
    acc = np.full(rest, _identity(op, d), NP_DTYPE[d])
    for k in range(flat.shape[0]):
        if op == "ADD":
            acc = _wrap(acc.astype(np.int64) + flat[k]) if d == "i32" else (acc + flat[k]).astype(np.float32)
        else:
            acc = _ieee_max(acc, flat[k]) if d == "f32" else np.maximum(acc, flat[k])
    return acc


def _reshape_rule(u):
    if prod(u.arg) != prod(u.src[0].shape):
        raise TypeError(f"cannot RESHAPE {list(u.src[0].shape)} to {list(u.arg)}")
    return u.src[0].dtype, tuple(u.arg)


TYPE_RULES.update({"REDUCE": _reduce_rule, "EXPAND": _expand_rule, "RESHAPE": _reshape_rule,
                   "STAGE": lambda u: (u.src[0].dtype, u.src[0].shape)})
EVAL.update({"REDUCE": _ev_reduce, "EXPAND": lambda u, x: np.broadcast_to(x, u.arg).copy(),
             "RESHAPE": lambda u, x: x.reshape(u.arg), "STAGE": lambda u, x: x})
BOUNDS_RULES.update({"EXPAND": lambda u: bounds(u.src[0]), "STAGE": lambda u: bounds(u.src[0]),
                     "RESHAPE": lambda u: bounds(u.src[0])})


def _unit_reshape(u, idx, ctx):
    """Week 4 RESHAPEs only add or remove size-1 axes; general reshapes arrive with movement_index."""
    keep = [i for i, n in zip(idx, u.shape) if n != 1]
    if [n for n in u.shape if n != 1] != [n for n in u.src[0].shape if n != 1]:
        raise NotImplementedError("only size-1 axes can be added or removed before week 5")
    keep.reverse()
    return tuple(const(0) if n == 1 else keep.pop() for n in u.src[0].shape)


def flat_index(idx, shape):
    e = const(0)
    for i, s in zip(idx, strides(shape)):
        e = UOp("ADD", (e, UOp("MUL", (i, const(s)))))
    return e


MOVE_INDEX = {
    "INDEX": lambda u, idx, ctx: tuple(lower_index(i, (), ctx) for i in u.src[1:]) + tuple(idx),
    "SHRINK": lambda u, idx, ctx: tuple(UOp("ADD", (idx[d], lower_index(b, (), ctx))) for d, b in enumerate(u.src[1:]))
    + tuple(idx[len(u.src) - 1:]),
    "EXPAND": lambda u, idx, ctx: tuple(const(0) if n == 1 and m != 1 else i
                                        for i, n, m in zip(idx, u.src[0].shape, u.shape)),
    "RESHAPE": _unit_reshape,
}
LOWER_RULES = {}


class Lowering:
    def __init__(self):
        self.memo, self.mat, self.eff = {}, {}, {}
        self.slots = itertools.count(1 << 24)

    def fresh_ranges(self, shape):
        return tuple(const(0) if n == 1 else UOp("RANGE", (const(n),), next(self.slots)) for n in shape)

    def close(self, eff, rs):
        for r in reversed(rs):
            if r.op == "RANGE":
                eff = UOp("END", (eff, r))
        return eff

    def alloc(self, dtype, shape):
        return UOp("ALLOC", (), BufArg(next(self.slots), dtype, (max(1, prod(shape)),), None, None))

    def lower_effect(self, e):
        if e in self.eff:
            return self.eff[e]
        if e.op == "STORE":
            dst, val = e.src
            rs = self.fresh_ranges(val.shape)
            base, addr = view_address(dst, rs, self)
            out = self.close(UOp("STORE", (UOp("INDEX", (base, addr)), lower_index(val, rs, self))), rs)
        elif e.op == "END":
            out = UOp("END", (self.lower_effect(e.src[0]), e.src[1]))
        else:
            raise NotImplementedError(f"cannot lower the effect {e.op}")
        self.eff[e] = out
        return out

    def materialize(self, u):
        if u in self.mat:
            return self.mat[u]
        x = u.src[0]
        if u.op == "STAGE":
            buf = self.alloc(x.dtype, x.shape)
            rs = self.fresh_ranges(x.shape)
            store = UOp("STORE", (UOp("INDEX", (buf, flat_index(rs, x.shape))), lower_index(x, rs, self)))
            out = UOp("AFTER", (buf, self.close(store, rs)))
        elif u.op == "REDUCE":
            op, pop = u.arg
            kept = x.shape[pop:]
            buf = self.alloc(x.dtype, kept)
            r_rest = self.fresh_ranges(kept)
            r_red = self.fresh_ranges(x.shape[:pop])
            at = flat_index(r_rest, kept)
            init = UOp("STORE", (UOp("INDEX", (buf, at)), const(_identity(op, x.dtype), x.dtype)))
            red_ranges = tuple(r for r in r_red if r.op == "RANGE")
            acc = UOp("INDEX", (UOp("AFTER", (buf, init) + red_ranges), at))
            val = UOp(op, (UOp("LOAD", (acc,)), lower_index(x, r_red + r_rest, self)))
            update = UOp("STORE", (acc, val))
            out = UOp("AFTER", (buf, self.close(self.close(update, r_red), r_rest)))
        else:
            raise NotImplementedError(f"cannot materialize {u.op}")
        self.mat[u] = out
        return out


def view_address(v, idx, ctx):
    if v.op in ("BUFFER", "ALLOC"):
        base = v if len(v.shape) == 1 else UOp("RESHAPE", (v,), (prod(v.shape),))
        return base, flat_index(idx, v.shape)
    if v.op == "AFTER":
        base, addr = view_address(v.src[0], idx, ctx)
        deps = tuple(e if e.op == "RANGE" else ctx.lower_effect(e) for e in v.src[1:])
        return UOp("AFTER", (base,) + deps), addr
    if v.op in MOVE_INDEX:
        return view_address(v.src[0], MOVE_INDEX[v.op](v, idx, ctx), ctx)
    raise NotImplementedError(f"cannot address memory through {v.op}")


def lower_index(u, idx, ctx):
    idx = tuple(idx)
    key = (u, idx)
    if key in ctx.memo:
        return ctx.memo[key]
    op = u.op
    if op in ("CONST", "PARAM", "RANGE"):
        out = u
    elif op == "LOAD":
        base, addr = view_address(u.src[0], idx, ctx)
        out = UOp("LOAD", (UOp("INDEX", (base, addr)),))
    elif op in ("REDUCE", "STAGE"):
        buf = ctx.materialize(u)
        out = UOp("LOAD", (UOp("INDEX", (buf, flat_index(idx, u.shape))),))
    elif op == "STACK":
        i, rest = idx[0], idx[1:]
        if i.op == "CONST":
            out = lower_index(u.src[i.arg], rest, ctx)
        elif len(u.shape) == 1 and all(s.op == "CONST" for s in u.src):
            out = UOp("INDEX", (u, i))
        else:
            parts = [lower_index(s, rest, ctx) for s in u.src]
            out = parts[-1]
            for k in range(len(parts) - 2, -1, -1):
                out = UOp("WHERE", (UOp("CMPLT", (i, const(k + 1))), parts[k], out))
    elif op in LOWER_RULES:
        out = LOWER_RULES[op](u, idx, ctx)
    elif op in MOVE_INDEX:
        out = lower_index(u.src[0], MOVE_INDEX[op](u, idx, ctx), ctx)
    elif op in ALU_OPS:
        out = UOp(op, tuple(lower_index(s, idx, ctx) for s in u.src), u.arg)
    else:
        raise NotImplementedError(f"cannot lower {op}")
    ctx.memo[key] = out
    return out

# ── Step 022  rangeify ──
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


PRE_PASSES = []


def rangeify(root):
    root = graph_rewrite(root, [inline_calls])
    ctx = Lowering()
    if root.op == "AFTER":
        out = UOp("AFTER", (root.src[0],) + tuple(ctx.lower_effect(e) for e in root.src[1:]))
    elif root.dtype == "void":
        out = ctx.lower_effect(root)
    elif root.shape == ():
        out = lower_index(root, (), ctx)
    else:
        raise ValueError("a shaped result must be STOREd into a buffer")
    return graph_rewrite(out, [fold_constants, symbolic])

# ── Step 023  run_uir ──
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


URL_FETCHER = None


def run_uir(program, buffers=None, params=None):
    root = parse_uir(program) if isinstance(program, str) else program
    for p in PRE_PASSES:
        root = p(root)
    buffers = {} if buffers is None else buffers
    fetcher = globals().get("URL_FETCHER")
    if fetcher is not None:
        for u in root.toposort():
            if u.op == "BUFFER" and u.arg.url and u.arg.slot not in buffers:
                buffers[u.arg.slot] = fetcher(u.arg.url, u.arg.sha256, u.dtype, tuple(u.arg.shape))
    return run_program(rangeify(root), buffers, params)

# ── Step 024  movement_index ──
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


def _permute_rule(u):
    x, order = u.src[0], tuple(u.arg)
    if sorted(order) != list(range(len(x.shape))):
        raise TypeError(f"{list(order)} is not a permutation of {len(x.shape)} axes")
    return x.dtype, tuple(x.shape[o] for o in order)


def movement_index(u, idx, ctx):
    idx = tuple(idx)
    if u.op == "RESHAPE":
        src_shape = u.src[0].shape
        flat = flat_index(idx, u.shape)
        out = []
        for k, (dim, st) in enumerate(zip(src_shape, strides(src_shape))):
            e = flat if st == 1 else UOp("DIV", (flat, const(st)))
            if k != 0:
                e = UOp("MOD", (e, const(dim)))
            out.append(e)
        return tuple(out)
    if u.op == "PERMUTE":
        out = [None] * len(idx)
        for d, o in enumerate(u.arg):
            out[o] = idx[d]
        return tuple(out)
    raise NotImplementedError(f"movement_index does not handle {u.op}")


TYPE_RULES["PERMUTE"] = _permute_rule
EVAL["PERMUTE"] = lambda u, x: np.transpose(x, u.arg)
BOUNDS_RULES["PERMUTE"] = lambda u: bounds(u.src[0])
MOVE_INDEX.update({"RESHAPE": movement_index, "PERMUTE": movement_index})

# ── Step 025  pad_composite ──
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


def full(value, shape, dtype):
    c = const(value, dtype)
    return UOp("EXPAND", (UOp("RESHAPE", (c,), (1,) * len(shape)),), tuple(shape)) if shape else c


def _to_front(x, axis):
    order = (axis,) + tuple(d for d in range(len(x.shape)) if d != axis)
    inv = tuple(order.index(d) for d in range(len(order)))
    return (UOp("PERMUTE", (x,), order) if axis else x), inv


def pad_composite(x, pairs):
    for axis, (before, after) in enumerate(pairs):
        if before == 0 and after == 0:
            continue
        y, inv = _to_front(x, axis)
        n = y.shape[0]
        k = max(1, math.ceil(max(before, after) / n))
        zero = 0.0 if y.dtype == "f32" else (False if y.dtype == "bool" else 0)
        zeros = full(zero, y.shape, y.dtype)
        st = UOp("STACK", tuple([zeros] * k + [y] + [zeros] * k))
        flat = UOp("RESHAPE", (st,), ((2 * k + 1) * n,) + tuple(y.shape[1:]))
        y = UOp("SHRINK", (flat, const(k * n - before)), (n + before + after,))
        x = UOp("PERMUTE", (y,), inv) if axis else y
    return x

# ── Step 026  flip_composite ──
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


def flip_composite(x, axis=0):
    y, inv = _to_front(x, axis)
    n = y.shape[0]
    P = 1
    while P < n:
        P *= 2
    if P == 1:
        return x
    rest = tuple(y.shape[1:])
    if P != n:
        y = pad_composite(y, [(0, P - n)])
    b = P.bit_length() - 1
    y = UOp("RESHAPE", (y,), (2,) * b + rest)
    for j in range(b):
        t, jinv = _to_front(y, j)
        sw = UOp("STACK", (UOp("INDEX", (t, const(1))), UOp("INDEX", (t, const(0)))))
        y = UOp("PERMUTE", (sw,), jinv) if j else sw
    y = UOp("RESHAPE", (y,), (P,) + rest)
    y = UOp("SHRINK", (y, const(P - n)), (n,))
    return UOp("PERMUTE", (y,), inv) if axis else y

# ── Step 027  keccak_program ──
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


def keccak_constants():
    rho = [0] * 25
    x, y = 1, 0
    for t in range(24):
        rho[x + 5 * y] = ((t + 1) * (t + 2) // 2) % 64
        x, y = y, (2 * x + 3 * y) % 5
    lfsr, packed = 1, []
    for _ in range(24):
        bits = 0
        for j in range(7):
            bits |= (lfsr & 1) << j
            lfsr = ((lfsr << 1) ^ (0x71 if lfsr & 0x80 else 0)) & 0xFF
        packed.append(bits)
    return rho, packed


def keccak_program():
    rho, packed = keccak_constants()
    rc = [sum(((p >> j) & 1) << ((1 << j) - 1) for j in range(7)) for p in packed]

    src = [0] * 25
    for x in range(5):
        for y in range(5):
            src[y + 5 * ((2 * x + 3 * y) % 5)] = x + 5 * y
    rn = [(64 - rho[s]) % 64 for s in src]
    rn1 = [(v + 63) % 64 for v in rn]
    t0 = [s * 64 for s in src]
    t1 = [((s % 5 - 1) % 5) * 64 for s in src]
    t2 = [((s % 5 + 1) % 5) * 64 for s in src]
    n1 = [((L % 5 + 1) % 5 + 5 * (L // 5)) * 64 for L in range(25)]
    n2 = [((L % 5 + 2) % 5 + 5 * (L // 5)) * 64 for L in range(25)]
    rct = [(rc[r] >> z) & 1 for r in range(24) for z in range(64)]

    def tab(v):
        return "[" + " ".join(str(e) for e in v) + "]"

    return "\n".join([
        "%state = buffer 1600 : dtype=i32 slot=0",
        "%C = alloc 320 : dtype=i32 slot=1",
        "%B = alloc 1600 : dtype=i32 slot=2",
        "%r = range 24 : slot=0",
        "%st = after %state %r",
        "%i = range 320 : slot=1",
        "%c = mod (add (add (add (add (load (index %st %i)) (load (index %st (add %i 320)))) "
        "(load (index %st (add %i 640)))) (load (index %st (add %i 960)))) (load (index %st (add %i 1280)))) 2",
        "%ec = end (store (index %C %i) %c) %i",
        "%cv = after %C %ec",
        "%j = range 1600 : slot=2",
        "%lj = div %j 64",
        "%zj = mod %j 64",
        f"%zs = mod (add %zj (index {tab(rn)} %lj)) 64",
        f"%zs1 = mod (add %zj (index {tab(rn1)} %lj)) 64",
        f"%a = load (index %st (add (index {tab(t0)} %lj) %zs))",
        f"%d1 = load (index %cv (add (index {tab(t1)} %lj) %zs))",
        f"%d2 = load (index %cv (add (index {tab(t2)} %lj) %zs1))",
        "%eb = end (store (index %B %j) (mod (add (add %a %d1) %d2) 2)) %j",
        "%bv = after %B %eb",
        "%k = range 1600 : slot=3",
        "%lk = div %k 64",
        "%zk = mod %k 64",
        "%b0 = load (index %bv %k)",
        f"%b1 = load (index %bv (add (index {tab(n1)} %lk) %zk))",
        f"%b2 = load (index %bv (add (index {tab(n2)} %lk) %zk))",
        f"%io = where (cmplt %k 64) (index {tab(rct)} (add (mul %r 64) %zk)) 0",
        "%v = mod (add (add %b0 (cast (cmplt %b1 %b2) : i32)) %io) 2",
        "%ek = end (store (index %state %k) %v) %k",
        "%result = after %state (end %ek %r)",
    ]) + "\n"


def count_uops(root):
    return sum(1 for u in root.toposort() if u.op != "CONST" or u is root)

# ── Step 028  gradient ──
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


def _pad_rule(u):
    x, befores, sizes = u.src[0], u.src[1:], tuple(u.arg)
    if not sizes or len(befores) != len(sizes) or len(sizes) > len(x.shape):
        raise TypeError("PAD needs one before and one size per leading axis")
    for b in befores:
        _scalar_i32(b, "a PAD before")
    for n, d in zip(sizes, x.shape):
        if n < d:
            raise TypeError("PAD cannot shrink an axis")
    return x.dtype, sizes + x.shape[len(sizes):]


def _ev_pad(u, x, *befores):
    out = np.zeros(tuple(u.arg) + x.shape[len(u.arg):], x.dtype)
    sl = []
    for k, (b, n) in enumerate(zip(befores, x.shape)):
        b = int(b)
        if b < 0 or b + n > u.arg[k]:
            raise IndexError("PAD places the input outside the output")
        sl.append(slice(b, b + n))
    out[tuple(sl)] = x
    return out


def _clamp(j, lo, hi):
    low = UOp("MUL", (UOp("MAX", (UOp("MUL", (j, const(-1))), const(-hi))), const(-1)))
    return UOp("MAX", (low, const(lo)))


def _lower_pad(u, idx, ctx):
    x = u.src[0]
    src, valid = list(idx), None
    for d, b in enumerate(u.src[1:]):
        j = UOp("ADD", (idx[d], UOp("MUL", (lower_index(b, (), ctx), const(-1)))))
        ok = UOp("WHERE", (UOp("CMPLT", (j, const(0))), const(False), UOp("CMPLT", (j, const(x.shape[d])))))
        valid = ok if valid is None else UOp("WHERE", (valid, ok, const(False)))
        src[d] = _clamp(j, 0, x.shape[d] - 1)
    zero = const(_identity("ADD", x.dtype) if x.dtype != "bool" else False, x.dtype)
    return UOp("WHERE", (valid, lower_index(x, tuple(src), ctx), zero))


def _gradient_rule(u):
    loss, wrt = u.src
    if loss.dtype != "f32" or wrt.dtype != "f32":
        raise TypeError("GRADIENT needs an f32 loss and an f32 target")
    return "f32", wrt.shape


TYPE_RULES.update({"PAD": _pad_rule, "DETACH": lambda u: (u.src[0].dtype, u.src[0].shape), "GRADIENT": _gradient_rule})
EVAL.update({"PAD": _ev_pad, "DETACH": lambda u, x: x})
LOWER_RULES.update({"PAD": _lower_pad, "DETACH": lambda u, idx, ctx: lower_index(u.src[0], idx, ctx)})


def _reduce_axes(g, axes, shape):
    keep = tuple(d for d in range(len(shape)) if d not in axes)
    t = UOp("PERMUTE", (g,), tuple(axes) + keep) if tuple(axes) + keep != tuple(range(len(shape))) else g
    return UOp("RESHAPE", (UOp("REDUCE", (t,), ("ADD", len(axes))),), tuple(1 if d in axes else shape[d] for d in range(len(shape))))


def _vjp(u, g):
    op = u.op
    if u.dtype != "f32":
        return []
    if op == "ADD":
        return [(u.src[0], g), (u.src[1], g)]
    if op == "MUL":
        a, b = u.src
        return [(a, UOp("MUL", (g, b))), (b, UOp("MUL", (g, a)))]
    if op == "MAX":
        a, b = u.src
        z = full(0.0, u.shape, "f32")
        lt = UOp("CMPLT", (a, b))
        return [(a, UOp("WHERE", (lt, z, g))), (b, UOp("WHERE", (lt, g, z)))]
    if op == "WHERE":
        c, a, b = u.src
        z = full(0.0, u.shape, "f32")
        return [(a, UOp("WHERE", (c, g, z))), (b, UOp("WHERE", (c, z, g)))]
    if op == "RECIP":
        x = u.src[0]
        neg = UOp("MUL", (g, full(-1.0, u.shape, "f32")))
        return [(x, UOp("MUL", (UOp("MUL", (neg, u)), u)))]
    if op == "CAST":
        x = u.src[0]
        return [(x, g)] if x.dtype == "f32" else []
    if op == "STAGE":
        return [(u.src[0], g)]
    if op == "EXPAND":
        x = u.src[0]
        axes = tuple(d for d, (n, m) in enumerate(zip(x.shape, u.shape)) if n == 1 and m != 1)
        return [(x, _reduce_axes(g, axes, u.shape) if axes else g)]
    if op == "REDUCE":
        x = u.src[0]
        rop, pop = u.arg
        lead = (1,) * pop + tuple(x.shape[pop:])
        ge = UOp("EXPAND", (UOp("RESHAPE", (g,), lead),), tuple(x.shape))
        if rop == "ADD":
            return [(x, ge)]
        mx = UOp("EXPAND", (UOp("RESHAPE", (u,), lead),), tuple(x.shape))
        z = full(0.0, x.shape, "f32")
        return [(x, UOp("WHERE", (UOp("CMPLT", (x, mx)), z, ge)))]
    if op == "RESHAPE":
        x = u.src[0]
        return [(x, UOp("RESHAPE", (g,), tuple(x.shape)))]
    if op == "PERMUTE":
        x = u.src[0]
        inv = [0] * len(u.arg)
        for d, o in enumerate(u.arg):
            inv[o] = d
        return [(x, UOp("PERMUTE", (g,), tuple(inv)))]
    if op == "SHRINK":
        x, begins = u.src[0], u.src[1:]
        return [(x, UOp("PAD", (g,) + tuple(begins), tuple(x.shape[:len(begins)])))]
    if op == "PAD":
        x, befores = u.src[0], u.src[1:]
        return [(x, UOp("SHRINK", (g,) + tuple(befores), tuple(x.shape[:len(befores)])))]
    if op == "STACK":
        return [(s, UOp("INDEX", (g, const(k)))) for k, s in enumerate(u.src)]
    if op == "INDEX":
        x, ids = u.src[0], u.src[1:]
        k = len(ids)
        r = UOp("RESHAPE", (g,), (1,) * k + tuple(g.shape))
        return [(x, UOp("PAD", (r,) + tuple(ids), tuple(x.shape[:k])))]
    return []


def gradient(loss, wrt):
    if loss.dtype != "f32" or wrt.dtype != "f32":
        raise TypeError("gradient needs an f32 loss and an f32 target")
    topo = loss.toposort()
    on = set()
    for u in topo:
        if u is wrt or any(id(s) in on for s in u.src):
            on.add(id(u))
    if id(wrt) not in on:
        return full(0.0, wrt.shape, "f32")
    adj = {id(loss): full(1.0, loss.shape, "f32")}
    for u in reversed(topo):
        key = id(u)
        if key not in adj or u is wrt or u.op == "DETACH":
            continue
        for x, gx in _vjp(u, adj[key]):
            if id(x) not in on:
                continue
            k = id(x)
            adj[k] = gx if k not in adj else UOp("ADD", (adj[k], gx))
    return adj[id(wrt)]

# ── Step 029  expand_gradients ──
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


def expand_gradients(root):
    root = graph_rewrite(root, [inline_calls])
    root = graph_rewrite(root, [lambda u: gradient(u.src[0], u.src[1]) if u.op == "GRADIENT" else None])
    return graph_rewrite(root, [lambda u: u.src[0] if u.op == "DETACH" else None])


PRE_PASSES.append(expand_gradients)

# ── Step 030  fetch_buffer ──
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


LITTLE_ENDIAN = {"f32": "<f4", "i32": "<i4", "bool": "|b1"}


def fetch_buffer(url, sha256, dtype, shape):
    with urllib.request.urlopen(url) as f:
        data = f.read()
    if sha256:
        digest = hashlib.sha256(data).hexdigest()
        if digest.lower() != str(sha256).lower():
            raise ValueError(f"sha256 mismatch for {url}: got {digest}")
    vals = np.frombuffer(data, dtype=LITTLE_ENDIAN[dtype])
    n = vals.size
    if n != prod(shape):
        raise ValueError(f"{url} holds {n} values, the BUFFER needs {prod(shape)}")
    return np.array(vals, dtype=NP_DTYPE[dtype], copy=True, order="C").reshape(tuple(shape))


URL_FETCHER = fetch_buffer

# ── Step 031  mnist_program ──
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

# ── Scaffold (runner) ──
from solution import *  # noqa: F401,F403

import time
import hashlib
import numpy as np

M32 = 0xFFFFFFFF

print("== Week 1: threefry2x32-20 with arithmetic only ==")
tf = parse_uir(threefry_program())
for key, ref in [((0, 0, 0, 0), (0x6B200159, 0x99BA4EFE)), ((M32,) * 4, (0x1CB996FC, 0xBB002BE7))]:
    got = tuple(int(v) & M32 for v in run_c(tf, dict(enumerate(key))))
    print("  key", [hex(k) for k in key], "->", [hex(v) for v in got], "matches" if got == ref else "MISMATCH")
print("  UOps:", count_uops(tf), "| ops:", " ".join(sorted({u.op for u in tf.toposort()})))

print("== Week 3: GEMM with explicit loops ==")
n = 128
rng = np.random.default_rng(0)
A, B = rng.standard_normal(n * n).astype(np.float32), rng.standard_normal(n * n).astype(np.float32)
gemm = parse_uir(gemm_program(n, n, n))
run_program(gemm, {0: A, 1: B})
t = time.time()
C = run_program(gemm, {0: A, 1: B})[2].reshape(n, n)
dt = time.time() - t
ref = np.zeros((n, n), np.float32)
for k in range(n):
    ref = (ref + A.reshape(n, n)[:, k:k + 1] * B.reshape(n, n)[k:k + 1, :]).astype(np.float32)
print(f"  {n}x{n}x{n}: bit-exact {np.array_equal(C, ref)}, {2 * n ** 3 / dt / 1e9:.2f} GFLOP/s (naive loops)")

print("== Week 5: Keccak-f[1600] ==")
kp = parse_uir(keccak_program())
for msg in (b"", b"abc"):
    block = bytearray(200)
    block[:len(msg)] = msg
    block[len(msg)] ^= 0x06
    block[135] ^= 0x80
    bits = np.unpackbits(np.frombuffer(bytes(block), np.uint8), bitorder="little").astype(np.int32)
    digest = np.packbits(run_uir(kp, {0: bits})[0].astype(np.uint8), bitorder="little").tobytes()[:32].hex()
    print(f"  SHA3-256({msg!r}) = {digest[:16]}... {'matches hashlib' if digest == hashlib.sha3_256(msg).hexdigest() else 'MISMATCH'}")
print("  UOps:", count_uops(kp), "(course budget: 99)")

print("== Week 6: MNIST trained by one compiled .uir program ==")
files = prepare_mnist(60000, 10000)
t = time.time()
res = run_uir(mnist_program(60000, 10000, files))
print(f"  784-32-10 MLP, 1 epoch of SGD, batch 16: test accuracy {int(res[5][0]) / 10000:.4f} in {time.time() - t:.1f} s")
