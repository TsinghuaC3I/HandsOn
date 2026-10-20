# G01 supplier assets

Source archive: `ZRYZ_LW_Assets_20260807.zip`

SHA-256: `90798bdad84d8b1de38eb625984f7fefa4ae3b56c7297fb4769bd35151189306`

The original directory layout is retained because the USD files use relative
texture and payload references. At runtime Pillow024's supplier tetrahedral
data is migrated to the current PhysX volume-deformable hierarchy. The thick,
closed Towl011 volume is retained only as a material/texture source and is
replaced by an E01-style single-mesh surface deformable so it can form cloth
folds instead of behaving like a rubber mouse pad. Powercable001 remains a
66-rigid-body joint chain. G01 therefore requires the Isaac Sim 5 deformable
beta pipeline; it is not supported by the Isaac Sim 6/Newton backend.

G01 is qualified for live inspection and collection interoperability. Formal
multi-deformable physical replay is intentionally reported as uncertified.
