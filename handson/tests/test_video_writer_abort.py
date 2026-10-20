from __future__ import annotations

from handson.data.video_writer import Mp4VideoWriter


def test_abort_closes_container_without_flushing_encoder() -> None:
    calls = []

    class Stream:
        def encode(self):
            raise AssertionError("abort must not flush delayed frames")

    class Container:
        def close(self):
            calls.append("close")

    writer = Mp4VideoWriter(
        fps=30,
        codec="h264",
        input_pix_fmt="rgb24",
    )
    writer._stream = Stream()
    writer._container = Container()
    writer._shape = (48, 64, 3)

    writer.abort()

    assert calls == ["close"]
    assert writer._stream is None
    assert writer._container is None
    assert writer._shape is None
