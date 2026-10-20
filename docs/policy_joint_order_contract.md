# Policy joint-order contract

PhysX/Isaac Lab stores robot joint tensors in the articulation's
`robot.joint_names` order. The Wuji controller consumes its 20 hand joints in
finger-major order. These orders are different and must never be inferred from
array shape alone.

For every PhysX recording, `recording_schema.json` is the source of truth:

- `environment_schema.physx.robot_joint_names` defines the raw joint axis.
- `environment_schema.physx.robot_joint_order_schema` identifies that axis as
  `handson.physx.articulation_joint_names.v1` in new recordings.
- `environment_schema.physx.policy_joint_orders` stores the Wuji hand20 names
  and their articulation indices.

Dataset conversion must resolve joints by name with
`handson.policy_joint_order`, not by slicing `[..., 7:27]`. A canonical EEF52
dataset stores the returned `wuji_hand_order_contract` in all three locations:

- `meta/info.json` under `handson_policy_schema`;
- `meta/handson_policy_schema.json`;
- the normalization asset directory as `handson_policy_schema.json`, so the
  contract is copied into checkpoints.

The canonical schema ID is `handson.policy.actual_eef52.v1`; its hand target
order is `handson.wuji.hand20.control.v1`. Missing markers mean **unknown**, not
"probably converted". Training and evaluation launchers should run:

```bash
python openpi/scripts/audit_handson_policy_schema.py DATASET [CHECKPOINT]
```

The command exits nonzero for unmarked data, malformed permutations, or a
dataset/checkpoint schema mismatch. Legacy data may only be stamped after an
explicit audit of its converter provenance; directory names are not evidence.
