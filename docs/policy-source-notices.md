# Native policy sources

The model integrations include the source used by the paper deployment, with
portable imports, explicit asset paths and the original inference math.
`handson/handson/policies/source_provenance.json` records the original and
migrated SHA256 digests. Local training launchers, credentials, dataset readers
in the custom adapters, and experiment-specific absolute paths are excluded.
Upstream helper modules needed by native models retain their original license
headers. This repository's root license does not replace third-party licenses.

| Source | Location under `policies/vendor/` | License |
| --- | --- | --- |
| OpenPI and the paper π0.5 implementation | `pi05`, `openpi_client` | Apache-2.0; see retained OpenPI headers and `pi05/LICENSE` |
| HY-Embodied-0.5-VLA | `hy` | Apache-2.0; Tencent notice retained |
| OpenDM | `dm05` | Apache-2.0 |
| LingBot-VLA and its LeRobot dependency | `lingbot`, `lingbot_lerobot` | Apache-2.0 |
| LeRobot diffusion policy | `dp` | Apache-2.0 |
| OpenWAM | `openwam` | Apache-2.0; nested Wan/DiffSynth headers retained |
| [FastWAM](https://github.com/yuantianyuan01/FastWAM) | `fastwam` | MIT, except files with their own third-party notices |
| [Motus](https://github.com/thu-ml/Motus), Wan and Qwen vision utilities | `motus` | Apache-2.0; component headers retained |
| [Spatial Forcing](https://github.com/OpenHelix-Team/Spatial-Forcing), including its OpenPI implementation | `spatial` | MIT for Spatial Forcing; inherited OpenPI headers remain Apache-2.0 |

The custom `adapters/` and `native/` integrations preserve the model architectures,
selected checkpoint steps, normalization, image processing, random-noise streams
and action prefix lengths. Changes remove private filesystem layout and unused
training entry points, extract inference-only helpers, and unify the external
HTTP boundary. The precise upstream source snapshots are identified by file
hashes rather than a claim that an arbitrary current upstream checkout is
equivalent. Foundation weights and checkpoints have their own distribution
terms and are not included here.
