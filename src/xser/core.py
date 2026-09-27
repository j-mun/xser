"""
```
Author: Jungho Mun
Date: 2026-01-31
THIS FILE IS PART OF XSER PACKAGE
```
"""

from __future__ import annotations

__all__ = [
  'SpecNode',
  'to_spec',
  'from_spec',
  'from_json',
]

from typing import Any
import base64
import importlib
import inspect
import json
from typing import ClassVar
from dataclasses import dataclass, fields, replace as _replace

import jax
from jax.tree_util import register_pytree_node_class

import jax.numpy as jp
import numpy as np

Array = jp.ndarray | np.ndarray

KEY = '__key__'
SPECNODE = '__xsn__'
UNRECORDABLE = '__invalid__'
ARRAY = '__arr__'


_SPECNODE_REGISTRY:dict[str,type] = {}
_SPECNODE_BY_NAME:dict[str,dict[str,type]] = {}

def is_prng_key(leaf:Any)->bool:
  return hasattr(leaf, 'dtype') and jp.issubdtype(leaf.dtype, jax.dtypes.prng_key)

def specnode_tag(cls:type)->str:
  return f'{cls.__module__}:{cls.__name__}'

def register_specnode_spec(cls:type)->type:
  tag = specnode_tag(cls)
  _SPECNODE_REGISTRY[tag] = cls
  _SPECNODE_BY_NAME.setdefault(cls.__name__, {})[tag] = cls
  return cls

def resolve_specnode(tag:str)->type:
  if not isinstance(tag, str):
    raise TypeError()
  if tag in _SPECNODE_REGISTRY:
    return _SPECNODE_REGISTRY[tag]

  module,_,name = tag.rpartition(':')
  candidates = _SPECNODE_BY_NAME.get(name or tag, {})
  if len(candidates) == 1:
    return next(iter(candidates.values()))
  if len(candidates) > 1:
    raise ValueError(f"{tag!r} is ambiguous by name; candidates are {sorted(candidates)}")

  if not module:
    raise ValueError(f"unknown {SPECNODE} tag {tag!r} and no module to import")
  try:
    return getattr(importlib.import_module(module), name)
  except (ImportError, AttributeError) as exc:
    raise ValueError(f"cannot resolve {SPECNODE} tag {tag!r}: {exc}") from exc

#--- serialization

def _to_spec(value:Any)->Any:
  '''convert a value to a serializable specification'''

  if value is None:
    return None

  # custom serialization for objects with a `to_spec` method
  if hasattr(value, "to_spec") and callable(value.to_spec):
    return value.to_spec()

  # dict
  if isinstance(value, dict):
    return {k: _to_spec(v) for k,v in value.items()}

  # list or tuple
  if isinstance(value, (list,tuple)):
    return type(value)(_to_spec(v) for v in value)

  # primitive types
  if isinstance(value, (bool,int,float,str)):
    return value

  # prng keys
  if is_prng_key(value):
    data = np.asarray(jax.random.key_data(value))
    return {KEY : base64.b64encode(data.tobytes()).decode('ascii'),
           'dtype': str(data.dtype),
           'shape': data.shape}

  # numpy or jax arrays
  if isinstance(value, (jp.ndarray, np.ndarray)):
    raw = np.asarray(value).tobytes()
    return {
      ARRAY   : base64.b64encode(raw).decode("ascii"),
      "dtype" : str(np.asarray(value).dtype),
      "shape" : np.asarray(value).shape
    }

  # fallback for unrecordable types
  described = getattr(value, "__name__", None) or type(value).__name__

  return {UNRECORDABLE: described}

to_spec = _to_spec






#--- deserialization
def from_spec(spec:Any)->Any:

  if isinstance(spec, dict):
    if SPECNODE in spec:
      return resolve_specnode(spec[SPECNODE]).from_spec(spec)

    if KEY in spec:
      payload = spec[KEY]
      if isinstance(payload, str):
        payload = base64.b64decode(payload.encode('ascii'))
      data = np.frombuffer(payload, dtype=spec['dtype']).reshape(spec['shape'])
      return jax.random.wrap_key_data(jp.asarray(data))
    
    if ARRAY in spec:
      payload = spec[ARRAY]
      if isinstance(payload, str):
        payload = base64.b64decode(payload.encode("ascii"))
      return jp.asarray(np.frombuffer(payload, dtype=spec["dtype"]).reshape(spec["shape"]))

    if UNRECORDABLE in spec:
      raise ValueError(f"cannot deserialize {spec[UNRECORDABLE]}")
    return {k: from_spec(v) for k,v in spec.items()}

  if isinstance(spec, (list,tuple)):
    return type(spec)(from_spec(v) for v in spec)

  return spec

def from_json(filename:str)->Any:
  with open(filename, "r") as f:
    spec = json.load(f)
  return from_spec(spec)








class SpecNode:
  '''Serializable Pytree Node'''
  _data:ClassVar[tuple[str,...]] = ()
  _meta:ClassVar[tuple[str,...]] = ()

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    dataclass(frozen=True)(cls)
    if not inspect.isabstract(cls):
      register_pytree_node_class(cls)
    declared = set(cls._data) | set(cls._meta)
    actual = {f.name for f in fields(cls)}
    if declared != actual:
      raise TypeError()

  #::pytree protocol
  def tree_flatten(c):
    return (tuple(getattr(c,k) for k in c._data),
            tuple(getattr(c,k) for k in c._meta))
  @classmethod
  def tree_unflatten(cls, meta, data):
    obj = object.__new__(cls)
    for k,v in zip(cls._data,data):
      object.__setattr__(obj,k,v)
    for k,v in zip(cls._meta,meta):
      object.__setattr__(obj,k,v)
    return obj

  #::serialization
  def to_spec(c)->dict:
    return {
      SPECNODE: specnode_tag(type(c)),
      **{k: _to_spec(getattr(c,k)) for k in (c._data+c._meta)},
    }
  
  @classmethod
  def from_spec(cls, spec:dict):
    if not isinstance(spec, dict):
      raise TypeError()
    tag = spec.get(SPECNODE)
    if tag is None:
      raise ValueError()

    klass = resolve_specnode(tag)
    if not (isinstance(klass, type) and issubclass(klass, SpecNode)):
      raise ValueError()
    if klass is not cls:
      return klass.from_spec(spec)

    known = set(cls._data) | set(cls._meta)
    kwargs = {}
    for k,v in spec.items():
      if k not in known:
        continue
      v = from_spec(v)
      kwargs[k] = tuple(v) if isinstance(v, list) and k in cls._meta else v
    return cls(**kwargs)

  def to_json(c, filename:str)->None:
    spec = c.to_spec()
    with open(filename, "w") as f:
      json.dump(spec, f, indent=2)

  def is_equal(c, other: SpecNode) -> bool:
    if not isinstance(other, SpecNode):
      return False
    return c.to_spec() == other.to_spec()

  def replace(c, **kwargs):
    return _replace(c, **kwargs)

  def arr_fields(c):
    return fields(jax.Array)