# FireRedChat turn detector plugin for LiveKit Agents

This plugin provides FireRedChat's open-weight end-of-turn detection model for LiveKit Agents.

## Installation

```bash
pip install -e .
```

## Usage

### Chinese model

```python
from livekit.plugins.fireredchat_turn_detector.base import ChineseModel

session = AgentSession(
    ...
    turn_detection=ChineseModel(unlikely_threshold=0.08),
)
```

### Bilingual model

Bilingual model that currently supports the following languages: `English, Chinese`

```python
from livekit.plugins.fireredchat_turn_detector.base import MultilingualModel

session = AgentSession(
    ...
    turn_detection=MultilingualModel(unlikely_threshold=0.08),
)
```

## License

The plugin source code and model weights are licensed under the Apache-2.0 license.
