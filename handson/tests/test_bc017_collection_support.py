from __future__ import annotations

from handson.runtime import collection_support


BC017_TASK_NAME = "t15_open_drawers"


def test_bc017_is_an_ordinary_collection_task() -> None:
    assert BC017_TASK_NAME in collection_support.ISAACSIM_COLLECTION_TASKS
    assert BC017_TASK_NAME not in (
        collection_support.ISAACSIM_COLLECTION_UNVERIFIED_TASKS
    )
    assert BC017_TASK_NAME in collection_support.ISAACSIM_FORMAL_COLLECTION_TASKS


def test_bc017_has_no_release_capability_gate() -> None:
    assert not hasattr(collection_support, "BC017_RELEASE_PINNED")
    assert not hasattr(collection_support, "BC017_RUNTIME_FORMAL")
    assert not hasattr(collection_support, "_bc017_formal_capability")
