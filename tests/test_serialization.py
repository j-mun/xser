from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

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


def assert_affine_equal(actual: Affine, expected: Affine) -> None:
	assert isinstance(actual, Affine)
	assert actual.activation == expected.activation
	np.testing.assert_array_equal(actual.weight, expected.weight)
	np.testing.assert_array_equal(actual.bias, expected.bias)


def make_affine() -> Affine:
	return Affine(
		weight=jp.asarray([[1.0, -2.0], [0.5, 3.0]]),
		bias=jp.asarray([0.25, -1.0]),
		activation='relu',
	)


def test_custom_spec_node_is_jax_compatible():
	affine = make_affine()
	values = jp.asarray([[2.0, -1.0], [-3.0, 4.0], [0.5, 0.5]])

	expected = jp.stack([affine(value) for value in values])
	jitted = jax.jit(lambda node, value: node(value))(affine, values[0])
	vmapped = jax.vmap(lambda node, value: node(value), in_axes=(None, 0))(
		affine, values
	)

	np.testing.assert_allclose(jitted, expected[0])
	np.testing.assert_allclose(vmapped, expected)


def test_custom_spec_node_spec_round_trip():
	affine = make_affine()

	spec = affine.to_spec()
	assert spec == xser.to_spec(affine)

	restored = xser.from_spec(spec)
	assert_affine_equal(restored, affine)


def test_custom_spec_node_json_round_trip(tmp_path: Path):
	affine = make_affine()
	filename = tmp_path / 'affine.json'

	affine.to_json(filename)

	assert filename.is_file()
	restored = xser.from_json(filename)
	assert_affine_equal(restored, affine)
