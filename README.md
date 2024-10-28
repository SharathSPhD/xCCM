# README.md
```markdown
# Extended CCM

An extended implementation of Convergent Cross Mapping (CCM) building upon the skccm package.

## Features

- Enhanced CCM implementation with signal decomposition
- Multivariate analysis capabilities
- Network analysis tools
- Advanced visualization
- Performance optimization
- Comprehensive testing suite

## Installation

```bash
pip install xCCM
```

## Quick Start

```python
from xCCM import EnhancedCCM, CCMVisualizer

# Initialize CCM
ccm = EnhancedCCM(embedding_dimension=3)

# Run analysis
results = ccm.run_ccm(x, y, lib_sizes)

# Visualize results
viz = CCMVisualizer()
viz.plot_convergence(lib_sizes, [r.correlation for r in results])
```

## Documentation

See the `examples/` directory for detailed usage examples.

## Development

To set up the development environment:

```bash
# Clone the repository
git clone https://github.com/SharathSPhD/xCCM.git
cd xCCM

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install development dependencies
pip install -e ".[dev]"
```

## Testing

Run the test suite:

```bash
pytest
```
