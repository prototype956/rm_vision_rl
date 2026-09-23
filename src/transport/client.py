"""通过带长度前缀的本地协议访问仿真进程，供上层环境组织步进。"""
from __future__ import annotations

import json
import socket
import struct
from pathlib import Path

MAX_MESSAGE_BYTES = 16 * 1024 * 1024


class TrainingClient:
    """管理仿真请求编号及待确认请求，使用 Unix socket 同步通信。"""
    def __init__(self, path: str | Path, timeout: float = 120):
        self.path = str(path)
        self.timeout = timeout
        self.stream: socket.socket | None = None
        self.request_id = 0
        self.pending: dict | None = None
        self.round_id = 0
        self.step_id = 0
        self.connect()

    def connect(self):
        self.disconnect()
        self.stream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.stream.settimeout(self.timeout)
        self.stream.connect(self.path)

    def disconnect(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def _read_exact(self, size: int) -> bytes:
        result = bytearray()
        while len(result) < size:
            chunk = self.stream.recv(size - len(result))
            if not chunk:
                raise ConnectionError("simulator disconnected before complete response")
            result.extend(chunk)
        return bytes(result)

    def exchange(self, request: dict) -> dict:
        encoded = json.dumps(request, allow_nan=False, separators=(",", ":")).encode()
        if not 0 < len(encoded) <= MAX_MESSAGE_BYTES:
            raise ValueError("request exceeds framing limit")
        self.stream.sendall(struct.pack("!I", len(encoded)) + encoded)
        size, = struct.unpack("!I", self._read_exact(4))
        if not 0 < size <= MAX_MESSAGE_BYTES:
            raise ValueError("invalid response length")
        response = json.loads(self._read_exact(size))
        if response["version"] != 1 or response["request_id"] != request["request_id"]:
            raise ValueError(f"response correlation mismatch: {response}")
        return response

    def request(self, op: str, **fields) -> dict:
        if self.pending is not None:
            raise RuntimeError("previous response is unresolved; reconnect and retry_pending first")
        self.request_id += 1
        self.pending = dict(version=1, request_id=self.request_id, op=op, **fields)
        return self.retry_pending()

    def retry_pending(self) -> dict:
        """使用原请求编号重新获取待确认响应，避免通信恢复时创建重复操作。"""
        if self.pending is None:
            raise RuntimeError("no unresolved request")
        response = self.exchange(self.pending)
        self.pending = None
        if not response["ok"]:
            raise RuntimeError(response["error"])
        self.round_id = response["round_id"]
        self.step_id = response["step_id"]
        return response

    def reset(self, seed: int, scenario: dict | None = None) -> dict:
        return self.request("reset", seed=seed, scenario=scenario or {})

    def advance(self, *, yaw_rad=0.0, pitch_rad=0.0, distance_m=4.0,
                valid=True, fire=False) -> dict:
        """提交控制命令并请求下一控制步；角度单位为 rad，距离单位为 m。"""
        return self.request("advance", round_id=self.round_id, step_id=self.step_id + 1,
                            command=dict(valid=valid, yaw_rad=yaw_rad, pitch_rad=pitch_rad,
                                         distance_m=distance_m, fire=fire))

    def inspect(self) -> dict:
        return self.request("inspect")

    def end_window(self, max_settle_steps=1000) -> dict:
        """关闭当前射击窗口并开始自然结算，本请求不推进物理时间。"""
        return self.request("end_window", round_id=self.round_id, step_id=self.step_id,
                            max_settle_steps=max_settle_steps)

    def settle(self) -> dict:
        return self.request("settle", round_id=self.round_id, step_id=self.step_id + 1)

    def close(self):
        try:
            self.request("close")
        finally:
            self.disconnect()
