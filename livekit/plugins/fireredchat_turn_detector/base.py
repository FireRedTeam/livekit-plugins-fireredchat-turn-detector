from __future__ import annotations

import os
import asyncio
import json
import time
from abc import ABC, abstractmethod
import numpy as np
import re

from livekit.agents import llm
from livekit.agents.inference_runner import _InferenceRunner
from livekit.agents.ipc.inference_executor import InferenceExecutor
from livekit.agents.job import get_job_context

from .log import logger

MAX_HISTORY_TOKENS = 128

class _EUORunnerBase(_InferenceRunner):
    def __init__(self, lang="chinese"):
        super().__init__()
        self.lang = lang

    def initialize(self) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        try:
            self.tokenizer_path = os.path.join(os.path.dirname(__file__), "./pretrained_models/tokenizer")
            if self.lang == "chinese":
                self.local_path_onnx = os.path.join(os.path.dirname(__file__), "./pretrained_models/chinese_best_model_q8.onnx")
            elif self.lang == "multilingual":
                self.local_path_onnx = os.path.join(os.path.dirname(__file__), "./pretrained_models/multilingual_best_model_q8.onnx")
            else:
                raise NotImplementedError

            self._session = ort.InferenceSession(
                self.local_path_onnx, providers=["CPUExecutionProvider"]
            )

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path,
                local_files_only=True,
                truncation_side="left"
            )
        except OSError:
            raise RuntimeError(
                "livekit-plugins-firered-turn-detector initialization failed. "
            ) from None

    def run(self, data: bytes) -> bytes | None:
        data_json = json.loads(data)
        chat_ctx = data_json.get("chat_ctx", None)

        logger.info("eou start")
        if not chat_ctx:
            raise ValueError("chat_ctx is required on the inference input data")

        start_time = time.perf_counter()

        try:
            text = ""
            for msg in chat_ctx[::-1]:
                if msg["role"] == "user":
                    text = msg["content"] + text
                else:
                    break

            text = re.sub("[，。？！,. ?!]", "", text)

            logger.info(f"eou text: {text}")
            inputs = self._tokenizer(
                text,
                truncation=True,
                padding='max_length',
                add_special_tokens=False,
                return_tensors="np",
                max_length=MAX_HISTORY_TOKENS,
            )
            # Run inference
            outputs = self._session.run(None, 
                                        {
                                            "input_ids": inputs["input_ids"].astype("int64"), 
                                            "attention_mask": inputs["attention_mask"].astype("int64")
                                        })
            eou_probability = self.softmax(outputs[0]).flatten()[-1]
        except:
            eou_probability = 0.0

        end_time = time.perf_counter()

        data = {
            "eou_probability": float(eou_probability),
            "input": text,
            "duration": round(end_time - start_time, 3),
        }
        logger.info(f"eou end, prob: {float(eou_probability)}")
        return json.dumps(data, ensure_ascii=False).encode()

    @staticmethod
    def softmax(x):
        """Softmax函数"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class EOUModelBase(ABC):
    def __init__(
        self,
        inference_executor: InferenceExecutor | None = None,
        unlikely_threshold: float | None = None
    ) -> None:
        self._executor = inference_executor or get_job_context().inference_executor
        if unlikely_threshold:
            self._unlikely_threshold = unlikely_threshold
        else:
            self._unlikely_threshold = 0.5

    @abstractmethod
    def _inference_method(self): ...

    async def unlikely_threshold(self, language: str | None) -> float | None:
        return self._unlikely_threshold

    async def supports_language(self, language: str | None) -> bool:
        return True

    async def predict_eou(self, chat_ctx: llm.ChatContext) -> float:
        return await self.predict_end_of_turn(chat_ctx)

    # our EOU model inference should be fast, 1 second is more than enough
    async def predict_end_of_turn(
        self, chat_ctx: llm.ChatContext, *, timeout: float | None = 1
    ) -> float:
        messages = []

        for item in chat_ctx.items:
            if item.type != "message":
                continue

            if item.role not in ("user", "assistant"):
                continue

            for cnt in item.content:
                if isinstance(cnt, str):
                    messages.append(
                        {
                            "role": item.role,
                            "content": cnt,
                        }
                    )
                    break

        messages = messages[-1:]

        json_data = json.dumps({"chat_ctx": messages}, ensure_ascii=False).encode()

        result = await asyncio.wait_for(
            self._executor.do_inference(self._inference_method(), json_data),
            timeout=timeout,
        )

        assert result is not None, "end_of_utterance prediction should always returns a result"

        result_json = json.loads(result.decode())
        logger.debug(
            "eou prediction",
            extra=result_json,
        )
        return result_json["eou_probability"]

class _EUORunnerChinese(_EUORunnerBase):
    INFERENCE_METHOD = "lk_end_of_utterance_chinese"

    def __init__(self):
        super().__init__(lang="chinese")

class ChineseModel(EOUModelBase):
    def __init__(self, *, unlikely_threshold: float | None = None):
        super().__init__(unlikely_threshold=unlikely_threshold)

    def _inference_method(self) -> str:
        return _EUORunnerChinese.INFERENCE_METHOD


class _EUORunnerMultilingual(_EUORunnerBase):
    INFERENCE_METHOD = "lk_end_of_utterance_multilingual"

    def __init__(self):
        super().__init__(lang="multilingual")

class MultilingualModel(EOUModelBase):
    def __init__(self, *, unlikely_threshold: float | None = None):
        super().__init__(unlikely_threshold=unlikely_threshold)

    def _inference_method(self) -> str:
        return _EUORunnerMultilingual.INFERENCE_METHOD

_InferenceRunner.register_runner(_EUORunnerChinese)
_InferenceRunner.register_runner(_EUORunnerMultilingual)
