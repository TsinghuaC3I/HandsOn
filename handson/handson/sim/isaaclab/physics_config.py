"""Shared PhysX buffer sizing for HandsOn's Isaac Lab scenes.

Collection and task-preview processes run one environment.  Keeping the
multi-environment capacities there forces PhysX to pin a large host-memory
mapping before the first reset, which can fail even when GPU memory is mostly
free.
"""

from __future__ import annotations


SINGLE_ENV_GPU_MAX_RIGID_CONTACT_COUNT = 2**20
SINGLE_ENV_GPU_MAX_RIGID_PATCH_COUNT = 5 * 2**15

# Preserve the previous preview capacity when callers explicitly request a
# tiled, multi-environment scene.
MULTI_ENV_GPU_MAX_RIGID_CONTACT_COUNT = 2**22
MULTI_ENV_GPU_MAX_RIGID_PATCH_COUNT = 2**21


def bounded_single_env_gpu_physx_kwargs() -> dict[str, int]:
    """Return the proven low-pinning PhysX profile for one complex scene.

    Isaac Lab's remaining defaults reserve very large found/lost-pair,
    aggregate, heap, collision-stack, particle-contact, and temporary buffers.
    Repeated process-owned previews can fail those locked system-memory
    allocations with ``NV_ERR_NO_MEMORY`` even while VRAM and ordinary RAM are
    available, after which PhysX reports Xid 31/CUDA 700.  These capacities
    retain ample headroom for one HandsOn environment while bounding every
    unrelated preallocation.  Soft-body contacts stay at their 2**20 default
    because G01-2's high-resolution self-colliding pillow uses that stream.
    """

    return {
        "gpu_max_rigid_contact_count": 2**18,
        "gpu_max_rigid_patch_count": 2**14,
        "gpu_found_lost_pairs_capacity": 2**16,
        "gpu_found_lost_aggregate_pairs_capacity": 2**16,
        "gpu_total_aggregate_pairs_capacity": 2**16,
        "gpu_collision_stack_size": 2**22,
        "gpu_heap_capacity": 2**22,
        "gpu_temp_buffer_capacity": 2**20,
        "gpu_max_num_partitions": 1,
        "gpu_max_soft_body_contacts": 2**20,
        "gpu_max_particle_contacts": 2**16,
    }


def gpu_rigid_buffer_capacities(num_envs: int = 1) -> tuple[int, int]:
    """Return ``(contact_count, patch_count)`` for an Isaac Lab scene."""

    count = int(num_envs)
    if count < 1:
        raise ValueError("num_envs must be at least 1")
    if count == 1:
        return (
            SINGLE_ENV_GPU_MAX_RIGID_CONTACT_COUNT,
            SINGLE_ENV_GPU_MAX_RIGID_PATCH_COUNT,
        )
    return (
        MULTI_ENV_GPU_MAX_RIGID_CONTACT_COUNT,
        MULTI_ENV_GPU_MAX_RIGID_PATCH_COUNT,
    )
