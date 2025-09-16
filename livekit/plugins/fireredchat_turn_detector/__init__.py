# Modification copyright 2025 FireRedTeam
# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FireRedChat turn detection for LiveKit Agents
"""

import os
from livekit.agents import Plugin

from .log import logger
from .version import __version__

__all__ = ["base", "__version__"]

HG_MODEL = "FireRedTeam/fireredchat_turn_detector"
MODEL_REVISION = "0.0.1"

class EOUPlugin(Plugin):
    def __init__(self):
        super().__init__(__name__, __version__, __package__, logger)
    
    def download_files(self) -> None:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="FireRedTeam/fireredchat-turn-detector",
            local_dir=os.path.join(os.path.dirname(__file__), "pretrained_models")
        )

Plugin.register_plugin(EOUPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False

