from typing import Tuple, Optional, Union, cast
from tinygrad.shape.int_tuple import Integer, compact_strides, canonicalize_strides, crd2idx, product, shape_div, flatten
from tinygrad.shape.symbolic import Variable
import functools
from collections import deque
from dataclasses import dataclass

@dataclass(frozen=True)
class View:
  shape: Union[Tuple, Integer]
  strides: Union[Tuple, Integer]

  def __len__(self) -> int:
    return len(self.shape) if isinstance(self.shape, tuple) else 1

  def __call__(self, crd) -> Integer:
    return crd2idx(crd, self.shape, self.strides)

  def __getitem__(self, i) -> 'View':
    if isinstance(self.shape, tuple) and isinstance(self.strides, tuple):
      return View(self.shape[i], self.strides[i])
    else:
      assert i == 0
      return self

  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def size(self) -> int: return product(self.shape)

  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def cosize(self) -> int: return self(self.size() - 1) + 1

  @property
  @functools.lru_cache(maxsize=None)
  def flat_shape(self):
    return tuple(product(s) for s in self.shape) if isinstance(self.shape, tuple) else self.shape

  @staticmethod
  @functools.lru_cache(maxsize=None)
  def create(shape:Union[Integer, Tuple], strides:Optional[Union[Integer, Tuple]]=None):
    strides = canonicalize_strides(shape, compact_strides(shape)) if strides is None else strides
    return View(shape, strides)


  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def permute(self, perm: Tuple) -> 'View':
    assert isinstance(self.shape, tuple) and isinstance(self.strides, tuple)
    assert len(perm) == len(self.shape)
    new_shape = tuple(self.shape[i] for i in perm)
    new_strides = tuple(self.strides[i] for i in perm)
    return View.create(new_shape, new_strides)

  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def expand(self, new_shape: Tuple) -> 'View':
    if not isinstance(self.shape, tuple): raise ValueError(f"expand arg {self.shape=} must be a tuple")
    if len(new_shape) != len(self.shape): raise ValueError(f"expand arg {new_shape=} must have same number of dimensions as shape {self.shape=}")
    flat_shape = self.flat_shape
    if 0 in flat_shape:
      assert all((s == x == 0) or (s > 0 and (x % s) == 0) for s,x in zip(self.flat_shape, new_shape)), f"can't expand {self.shape} into {new_shape}"
      return View.create(new_shape)
    assert isinstance(self.strides, tuple)
    assert all((s == x or (s == 1 and st == 0)) for s,x,st in zip(flat_shape, new_shape, self.strides)), f"can't expand {self.shape} into {new_shape}"
    return View.create(new_shape, self.strides)


  def reshape(self, shape):
    assert product(self.shape) == product(shape), f"Cannot reshape {self.shape} to {shape}"
    return composition(self, View.create(shape))

  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def coalesce(self):
    result_shape  = deque([1])
    result_strides = deque([0])
    for (shape,strides) in zip(reversed(flatten(self.shape)),reversed(flatten(self.strides))):
      if shape == 1:
        continue
      elif result_shape[0] == 1:
        result_shape[0]  = shape
        result_strides[0] = strides
      elif result_shape[0] * result_strides[0] == strides:
        result_shape[0] = result_shape[0] * shape
      else:
        result_shape.appendleft(shape)
        result_strides.appendleft(strides)

    if len(result_shape) == 1:
      return View.create(result_shape[0], result_strides[0])
    else:
      return View.create(tuple(result_shape), tuple(result_strides))

  @property
  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def continuous(self) -> bool:
    v = self.coalesce()
    return v.strides == canonicalize_strides(v.shape, compact_strides(v.shape))

  def render(self):
    idxs: Union[Variable, Tuple[Variable, ...]]
    if isinstance(self.shape, tuple):
        idxs = tuple(Variable(f"idx{i}", 0, product(s)-1) for i,s in enumerate(self.shape))
    else:
        idxs = Variable("idx", 0, self.shape-1)
    return cast(Variable, self(idxs)).render()

  def __str__(self):
    return f"View(shape={self.shape}, strides={self.strides}, continuous={self.continuous})"

  def __repr__(self):
    return f"View(shape={self.shape}, strides={self.strides}, continuous={self.continuous})"

def make_view(*views: View) -> View:
  shape, strides = zip(*((a.shape,a.strides) for a in views))
  return View(shape, strides)

@functools.lru_cache(maxsize=None)
def composition(viewA: View, viewB:View):
  if viewB.strides == 0: return View.create(viewB.shape, 0)

  if isinstance(viewB.shape, tuple):
    return make_view(*tuple(composition(viewA, viewB[i]) for i in range(len(viewB.shape))))
  else:
    result_shape: deque[Union[int, Integer, Tuple]] = deque()
    result_strides: deque[Union[int, Integer, Tuple]]  = deque()
    rest_shape   = viewB.shape
    rest_strides  = viewB.strides
    for (s, d) in zip(reversed(flatten(viewA.shape)[1:]), reversed(flatten(viewA.strides)[1:])):
      s1 = shape_div(s, rest_strides)
      result_shape.appendleft(min(s1, rest_shape))
      result_strides.appendleft(rest_strides * d)
      rest_shape  = shape_div(rest_shape, abs(s1))
      rest_strides = shape_div(rest_strides, s)

    result_shape.appendleft(rest_shape)
    result_strides.appendleft(rest_strides * flatten(viewA.strides)[0])

    return View(tuple(result_shape), tuple(result_strides)).coalesce()
