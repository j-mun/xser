# XSER agent guidance

## Project shape

- The package lives under `src/xser/` and provides serializable JAX pytree nodes.
- [`src/xser/core.py`](src/xser/core.py) owns `SpecNode`, recursive spec conversion, JSON I/O, and type resolution.
- [`src/xser/__init__.py`](src/xser/__init__.py) is the public export surface.
- [`tests/test_serialization.py`](tests/test_serialization.py) contains the focused behavior tests.

## Development workflow

- The project requires Python `>=3.13`; install development dependencies with `pip install -e '.[dev]'`.
- Run the focused suite from the repository root with `python -m pytest -q`.
- Runtime dependencies are JAX, jaxlib, and NumPy; pytest is the development dependency.
- There is no configured formatter, linter, or type checker. Preserve the existing two-space package style and the test file's local indentation style.

## Implementation conventions

- `SpecNode` subclasses are frozen dataclasses and JAX pytree nodes. Keep `_data` for dynamic leaves and `_meta` for static auxiliary data.
- Every annotated dataclass field must appear exactly once in `_data` or `_meta`; the class setup validates this invariant.
- Keep wrappers and methods JAX-traceable. Avoid Python branching on traced array values; static behavior belongs in `_meta`.
- Static metadata should be immutable and hashable because JAX transformations use it as pytree auxiliary data.
- Serialize supported nested values through `to_spec`/`from_spec`; preserve array dtype and shape, PRNG-key handling, and `SpecNode` type tags.
- Define reusable `SpecNode` subclasses at module scope in importable modules so JSON deserialization can resolve their `module:class` tag in a new process.
- Keep public exports stable unless an API change is intentional. Do not add solver or optimizer logic here; sibling packages consume XSER as a serialization primitive.

## Validation cautions

- Test both JAX transformation behavior (`jit` and `vmap`) and spec/JSON round trips when changing `SpecNode` or serialization logic.
- Use `SpecNode.is_equal` or compare reconstructed fields/specs rather than relying on array `==` results directly.
- Check unknown, ambiguous, and unrecordable type handling when changing type resolution or recursive serialization.
- `pyproject.toml` declares an `xser = "xser.__main__:main"` console entry point, but this repository currently has no `src/xser/__main__.py`; do not assume the command exists without verifying it.

## Documentation map

- [`README.md`](README.md): installation, `SpecNode` usage, JAX transformations, and serialization API.
- [`pyproject.toml`](pyproject.toml): package metadata, supported Python version, dependencies, and entry points.