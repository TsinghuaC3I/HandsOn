"""Instance-owned transports for physical EEF52 policy commands."""
import http.client
import io
import json
import socket
import time
import urllib.error
import urllib.request

import numpy as np

from ..interfaces import validate_actions
from .contracts import require, write_new

RETRIES, RETRY_WAIT_S = 3, 5.0


class HoldTransport:
    def __init__(self, config):
        self.horizon = config.get("action_horizon", 30)

    def start(self):
        return dict(diagnostic_only=True, action_dim=52)

    def predict(self, state, images, index, seed):
        return np.repeat(np.asarray(state, dtype=np.float32)[None], self.horizon, axis=0)

    def close(self):
        pass


class HTTPTransport:
    def __init__(self, config):
        self.config = config
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def start(self):
        expected = self.config["required_metadata"]
        require(bool(expected) and any("checkpoint" in key for key in expected),
                "HTTP policy requires an explicit checkpoint identity")
        with self.opener.open(self.config["endpoint"], timeout=30) as response:
            return json.load(response)

    def predict(self, state, images, index, seed):
        c = self.config
        wire = c.get("front_wire_key", "front")
        require(wire in ("front", "base"), "Unknown policy image key")
        packet = io.BytesIO()
        np.savez(packet, state=state, **{wire: images["main"]},
                 wrist_left=images["wrist_left"], wrist_right=images["wrist_right"],
                 prompt=np.asarray(c["prompt"]), task=np.asarray(c.get("task", "")),
                 seed=np.asarray(seed), request=np.asarray(index))
        body = packet.getvalue()
        for attempt in range(RETRIES + 1):
            try:
                request = urllib.request.Request(c["endpoint"], data=body,
                    headers={"Content-Type": "application/octet-stream"})
                with self.opener.open(request, timeout=180) as response:
                    return json.load(response)["actions"]
            except urllib.error.HTTPError:
                raise
            except (urllib.error.URLError, ConnectionError, http.client.HTTPException,
                    socket.timeout, TimeoutError) as exc:
                if attempt == RETRIES:
                    raise
                print(json.dumps(dict(policy_retry=attempt + 1, request=index, error=repr(exc))), flush=True)
                time.sleep(RETRY_WAIT_S)

    def close(self):
        pass


class OpenPITransport:
    def __init__(self, config):
        self.config = config
        self.client = None

    def start(self):
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        self.client = WebsocketClientPolicy(host=self.config["host"], port=self.config["port"])
        return self.client.get_server_metadata()

    def predict(self, state, images, index, seed):
        c = self.config
        wire = c.get("front_wire_key", "base")
        return self.client.infer(dict(state=state, **{wire: images["main"]},
            wrist_left=images["wrist_left"], wrist_right=images["wrist_right"],
            prompt=c["prompt"]))["actions"]

    def close(self):
        close = getattr(self.client, "close", None)
        if close is not None:
            close()


TRANSPORTS = {"hold": HoldTransport, "http_physical_eef52": HTTPTransport,
              "openpi_websocket": OpenPITransport}


class PolicyClient:
    def __init__(self, config, output):
        self.config, self.output = config, output
        self.kind = config["transport"]
        require(self.kind in TRANSPORTS, "Unknown model transport")
        self.transport = TRANSPORTS[self.kind](config)

    def start(self):
        metadata = self.transport.start()
        expected = self.config.get("required_metadata", {})
        require(all(metadata.get(k) == v for k, v in expected.items()), "Model handshake mismatch")
        if self.kind != "hold":
            write_new(self.output / "READY.json", metadata)
        return metadata

    def predict(self, state, images, index, seed):
        # Preserve websocket/diagnostic evidence. HTTP observations are already
        # represented by the videos and steps.jsonl, without per-request files.
        if self.kind != "http_physical_eef52":
            np.savez(self.output / f"input_{index:03d}.npz", state=state,
                     front=images["main"], wrist_left=images["wrist_left"],
                     wrist_right=images["wrist_right"])
        actions = self.transport.predict(state, images, index, seed)
        actions = validate_actions(actions, self.config.get("action_horizon", 30))
        if self.kind == "openpi_websocket":
            np.save(self.output / f"actions_{index:03d}.npy", actions)
        return actions

    def close(self):
        self.transport.close()
