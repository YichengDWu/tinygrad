from collections import deque
import functools
from typing import Union, Tuple, Deque
from tinygrad.shape.symbolic import sint, Variable, MulNode, SumNode
from tinygrad.helpers import prod

Integer = sint  # might change
def is_int(x): return isinstance(x, (int, Variable, MulNode, SumNode))

@functools.lru_cache(maxsize=None)
def flatten(t):
  if isinstance(t, tuple):
    if len(t) == 0:
      return t
    else:
      return flatten(t[0]) + flatten(t[1:])
  else:
    return (t,)

@functools.lru_cache(maxsize=None)
def product(a): return prod(flatten(a)) if isinstance(a, tuple) else a
def signum(a): return bool(a > 0) - bool(a < 0)

@functools.lru_cache(maxsize=None)
def shape_div(a, b):
  if isinstance(a, tuple):    # tuple,
    r: Deque[Union[Integer, Tuple]] = deque()
    for v in reversed(a):
      r.appendleft(shape_div(v,b))
      b = shape_div(b, product(v))
    return tuple(r)
  else:
    if isinstance(b, tuple):                    # "int" tuple
      return shape_div(a, product(b))
    else:                              # "int" "int"
      assert a % b == 0 or b % a == 0, "shape_div: a and b are not divisible"
      if a % b == 0:
        return a // b
      else:
        return signum(a*b)

@functools.lru_cache(maxsize=None)
def compact_strides(a, init: Union[Tuple, int]=1) -> Union[Tuple, Integer]:
    if isinstance(a, tuple) and isinstance(init, tuple):
      assert len(a) == len(init)
      return tuple(compact_strides(v, i) for v, i in zip(a, init))
    if isinstance(a, tuple) and isinstance(init, int):
        r: Deque[Union[Integer, Tuple]] = deque()
        for v in reversed(a):
            r.appendleft(compact_strides(v, init))
            init = init * product(v)
        return tuple(r)
    if isinstance(init, int): return init
    raise RuntimeError("compact_strides: init is not an int")

@functools.lru_cache(maxsize=None)
def canonicalize_strides(shape:Union[Integer, Tuple], strides:Union[Integer, Tuple]) -> Union[Integer, Tuple]:
  if isinstance(shape, tuple) and isinstance(strides, tuple):
    return tuple(canonicalize_strides(s, d) for s, d in zip(shape, strides))
  elif is_int(shape) and is_int(strides):
    return 0 if shape == 1 else strides
  else:
    raise ValueError(f"Invalid combination of input values: {shape=}, {strides=}")

def idx2crd(idx, shape, strides=None):
    if strides is None:
        strides = compact_strides(shape)

    if is_int(idx) and is_int(shape) and is_int(strides):
        return (idx // strides) % shape
    elif is_int(idx) and isinstance(shape, tuple) and isinstance(strides, tuple):
      assert len(shape) == len(strides)
      return tuple(idx2crd(idx, s, d) for s,d in zip(shape,strides))
    elif isinstance(idx, tuple) and isinstance(shape, tuple) and isinstance(strides, tuple):
        if not (len(idx) == len(shape) == len(strides)):
            raise ValueError("Length of idx, shape, and strides must match for tuple inputs")
        return tuple(idx2crd(i, s, d) for i, s, d in zip(idx, shape, strides))
    else:
        raise TypeError("Invalid combination of input types")

def crd2idx(crd, shape, strides=None):
  if strides is None: strides = compact_strides(shape)

  if is_int(crd) and is_int(shape) and is_int(strides): return crd * strides
  elif isinstance(crd, tuple) and isinstance(shape, tuple) and isinstance(strides, tuple):
      assert len(crd) == len(shape) and len(crd) == len(strides)
      return sum(crd2idx(c, s, d) for c, s, d in zip(crd, shape, strides))
  elif is_int(crd) and isinstance(shape, tuple) and isinstance(strides, tuple):
    assert len(shape) == len(strides)
    result = 0
    for i in range(len(shape) - 1, 0, -1):
      result += crd2idx(crd % product(shape[i]), shape[i], strides[i])
      crd = crd // product(shape[i])
    return result + crd2idx(crd, shape[0], strides[0])
  else:
    raise ValueError(f"Invalid combination of input values: {crd=}, {shape=}, {strides=}")

def crd2crd(crd, dst_shape, src_shape):
  if isinstance(crd, tuple) and isinstance(dst_shape, tuple) and isinstance(src_shape, tuple):
      return tuple(crd2crd(x, y, z) for x, y, z in zip(crd, dst_shape, src_shape))
  else:
    return idx2crd(crd2idx(crd, src_shape), dst_shape)
