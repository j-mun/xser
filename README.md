# jaX SERialization (XSER)

`xser` provides serializable custom JAX pytrees. Subclass `xser.SpecNode`,
declare which fields are dynamic leaves and which are static metadata, and the
resulting class can be used with JAX transformations and saved as a portable
specification.

## Installation

```bash
pip install .
```

```bash
python -m pip install git+https://github.com/j-mun/xser.git
```


For development:

```bash
pip install -e '.[dev]'
pytest
```

## Defining a custom pytree

```python
import jax
import jax.numpy as jp
import xser


class Affine(xser.SpecNode):
	_data = ('weight', 'bias')
	_meta = ('activation',)

	weight: jax.Array
	bias: jax.Array
	activation: str = 'identity'

	def __call__(self, value: jax.Array) -> jax.Array:
		result = value @ self.weight + self.bias
		if self.activation == 'relu':
			return jp.maximum(result, 0)
		return result


layer = Affine(
	weight=jp.asarray([[1.0, -2.0], [0.5, 3.0]]),
	bias=jp.asarray([0.25, -1.0]),
	activation='relu',
)
```

`SpecNode` makes the subclass a frozen dataclass and registers it as a JAX
pytree node. Every annotated dataclass field must appear exactly once in one of
these class variables:

- `_data`: dynamic pytree leaves, such as parameters and arrays.
- `_meta`: static auxiliary data, such as modes, dimensions, and configuration.

Static metadata should be immutable and hashable when the object is passed to
JAX transformations.

## JAX transformations

Instances can be passed through `jax.jit` and `jax.vmap`:

```python
value = jp.asarray([2.0, -1.0])
result = jax.jit(lambda node, x: node(x))(layer, value)

batch = jp.asarray([[2.0, -1.0], [-3.0, 4.0]])
results = jax.vmap(
	lambda node, x: node(x),
	in_axes=(None, 0),
)(layer, batch)
```

## Serialization

Serialize to an in-memory specification and restore it with `xser.from_spec`:

```python
spec = layer.to_spec()
# Equivalent: spec = xser.to_spec(layer)

restored = xser.from_spec(spec)
```

Write JSON to a file and load it again:

```python
layer.to_json('affine.json')
restored = xser.from_json('affine.json')
```

Specifications support nested dictionaries, lists, tuples, primitive values,
NumPy arrays, JAX arrays, PRNG keys, and other `SpecNode` instances. Arrays and
keys are encoded with their dtype and shape so that they can be reconstructed.

The serialized pytree tag contains the subclass module and class name. Define
reusable subclasses at module scope in an importable module so they can be
resolved when loading JSON in a new Python process.

## API

- `xser.SpecNode`: base class for serializable JAX pytree nodes.
- `xser.to_spec(value)`: convert a supported value to a serializable spec.
- `xser.from_spec(spec)`: reconstruct a value from a spec.
- `node.to_json(filename)`: write a node's spec as JSON.
- `xser.from_json(filename)`: reconstruct a value from a JSON file.

