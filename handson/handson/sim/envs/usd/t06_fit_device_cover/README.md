# G02 migrated assets

These three USD files were copied byte-for-byte from hp-15 on 2026-08-14:

- source repository: `/mnt/pfs/luotianwei/dexverse-scenesmith`
- source scene: `outputs/isaaclab/scene_06_electronics_repair`
- source task: `repair_close_device`
- source branch: `dexverse-integration`
- source converter target: Isaac Lab 2.3.2

Only the static workbench and the two active task objects are included in
formal G02. The screwdriver, phone, charger, parts board, and pegs from the
scene preview are deliberately excluded from task dynamics and recording
truth. The source registry marks all three included assets as
`project_authored_no_external_asset_restrictions`.

HandsOn keeps the source meshes unchanged but lays the objects out in its
proven dual-arm pickup zones and assigns task-tuned hollow-enclosure masses and
high-friction rubber/plastic contacts for reliable Wuji manipulation.

SHA-256:

- `workbench/workbench.usd`: `2ec551ef0e7f98a30b08fe8ef1f7627e0a54e2b7fb4dad89b8333974527a1841`
- `device_body/device_body.usd`: `abb97c0efd39a25729a881151daaec7e70f6c3a0b62acb8aba0e46ec4addb4cf`
- `device_cover/device_cover.usd`: `02336b0355d5cdd9bcd801e4029b4c94f9799ec2996dcc8cb49e50988f25ef59`
