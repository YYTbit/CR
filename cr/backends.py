import os

import requests

from .storage import signature


class ApiBackend:
    def __init__(self, model, revision=None, endpoint="http://localhost:8000/v1/chat/completions",
                 key_variable="CR_API_KEY", timeout=30.0, max_tokens=128):
        if timeout <= 0 or max_tokens < 1:
            raise ValueError("Receiver endpoint timeout and completion length are positive")
        self.model, self.revision, self.endpoint = model, revision, endpoint
        self.key_variable, self.timeout, self.max_tokens = key_variable, float(timeout), int(max_tokens)
        self.session = requests.Session()
        self.data_kind = "measured_receiver"

    @property
    def identity(self):
        return {"kind": "api", "model": self.model, "revision": self.revision, "max_tokens": self.max_tokens,
                "endpoint_signature": signature(self.endpoint)}

    def complete(self, messages, temperature=0.0, seed=0, max_tokens=None):
        key = os.environ.get(self.key_variable)
        headers = {"Authorization": "Bearer " + key} if key else {}
        response = self.session.post(self.endpoint, headers=headers,
                                     json={"model": self.model, "messages": messages, "temperature": temperature,
                                           "seed": seed, "max_tokens": int(max_tokens or self.max_tokens)}, timeout=self.timeout)
        response.raise_for_status()
        record = response.json()
        return {"text": record["choices"][0]["message"]["content"], "usage": record.get("usage", {})}

    def close(self):
        self.session.close()
