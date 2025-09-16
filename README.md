# EchoLiner Labs

EchoLiner Labs provides open-source SDKs and reference implementations for AI-native modular manufacturing. This repository hosts the first public release with core modules for vision, robotics, translation, and analytics.

## Features

- **Vision**: Sobel edge detection using pure NumPy.
- **Robotics**: 2D planar arm model with forward kinematics.
- **Translation**: Simple offline English ↔ Chinese word translator.
- **Analytics**: Basic uptime and sensor statistics.

## Installation

```bash
pip install -e .[test]
```

## Usage

```python
from echoliner.vision.edge_detection import sobel_edges
from echoliner.robotics.kinematics import Arm2D
from echoliner.translation.simple_translator import translate
from echoliner.analytics.metrics import uptime_ratio
```

## Testing

Run the unit tests with:

```bash
pytest
```

## License

[MIT](LICENSE)
