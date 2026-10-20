"""Scene metadata for the D06--D19 long-horizon collection tasks.

The fourteen tasks share one MJCF asset library.  ``task_preview`` expands the
include and calls :func:`apply_d_long_task_filter` before converting the scene
to USD, so each cached task contains only its own objects and fixtures.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from . import d20_microwave_hot_dog as _d20
from . import d20_stages as _d20_stages
from . import d08_realistic_variant as _d08_variant


D_LONG_SCENE_XML = "arena_arm_hand_d_long_horizon_rokae_wuji1.xml"

D_LONG_TASK_SCENES = {
    "t42_sort_objects": D_LONG_SCENE_XML,
    "d07_organize_office_desk": D_LONG_SCENE_XML,
    "d08_toast_bread": D_LONG_SCENE_XML,
    "d08_2_realistic_toast": D_LONG_SCENE_XML,
    "t31_prepare_coffee": D_LONG_SCENE_XML,
    "d10_load_dishwasher": D_LONG_SCENE_XML,
    "d11_presoak_pan": D_LONG_SCENE_XML,
    "d12_prepare_cutting_station": D_LONG_SCENE_XML,
    "d13_prepare_tea_tray": D_LONG_SCENE_XML,
    "d14_wash_fruit": D_LONG_SCENE_XML,
    "d15_pack_matching_lunches": D_LONG_SCENE_XML,
    "d16_sweep_blocks_to_bin": D_LONG_SCENE_XML,
    "d17_store_laptop_headphones": D_LONG_SCENE_XML,
    "d18_pack_eggs": D_LONG_SCENE_XML,
    "d19_memory_cover_sort": D_LONG_SCENE_XML,
    "d20_microwave_hot_dog_plating": D_LONG_SCENE_XML,
}

D_LONG_TASK_SCENES.update({name: D_LONG_SCENE_XML for name in _d20_stages.STAGE_TASKS})
D_LONG_TASK_NAMES = frozenset(D_LONG_TASK_SCENES)

D_LONG_SUCCESS_CONDITION_LABELS = {
    "t42_sort_objects": (
        "Six target objects are in their correct category baskets",
        "No target object is in a wrong basket",
        "The far-right target completed a physical two-hand handoff",
        "All three baskets remain upright",
        "Neither hand touches any target for 0.5 seconds",
    ),
    "d07_organize_office_desk": (
        "Mouse, keyboard, and figurine reach their goal regions",
        "The alarm clock and all three office items are inside the drawer",
        "The drawer is fully closed",
        "No required object has fallen from the desk",
    ),
    "d08_toast_bread": (
        "Insert both bread slices fully into separate toaster slots",
        "Press the front button only after both slices are inserted",
        "Wait until the toast cycle finishes and both slices pop up",
        "After popping, remove both slices and place both on the plate",
        "Keep the toaster within 4 cm of its starting position",
        "Keep both slices stable on the plate and both hands away for 0.5 s",
    ),
    "t31_prepare_coffee": (
        "1. Open either cabinet door >= 0.65 rad",
        "2. Remove mug after opening a cabinet door",
        "3. Close both cabinet doors <= 0.10 rad",
        "4. Mug upright under outlet: XY <= 40 mm, base Z +/- 30 mm",
        "5. Press bright-green start button >= 1.5 mm",
        "6. Keep mug aligned for uninterrupted 2.0 s brew",
        "7. Place filled mug upright on serving tray",
        "8. Mug stable on tray; move both hands clear",
    ),
    "d10_load_dishwasher": (
        "Mug, bowl, and plate are stable in their assigned rack slots",
        "The upper rack is fully pushed in",
        "The dishwasher door is fully closed",
    ),
    "d11_presoak_pan": (
        "The pan and sponge are both inside the sink",
        "One hand keeps the pan stable while the faucet is open",
        "The faucet outlet covers the required pan regions for long enough",
        "The pan remains inside the sink",
    ),
    "d12_prepare_cutting_station": (
        "The drawer is closed and the cutting board is aligned",
        "The meat is centered on the cutting board",
        "The knife is fully inside the safety zone",
        "The knife blade points toward the prescribed outer direction",
    ),
    "d13_prepare_tea_tray": (
        "Kettle, mug, and tea bag occupy their marked tray regions",
        "The cabinet was opened and is finally closed",
        "The tray reaches the service region with peak tilt at most 15 degrees",
        "No payload fell from the tray during transport",
    ),
    "d14_wash_fruit": (
        "All three fruits were placed in the colander and covered by water",
        "The faucet is finally switched off",
        "The colander is stable in the draining region",
        "No fruit fell from the colander",
    ),
    "d15_pack_matching_lunches": (
        "The two lunch boxes contain identical food classes and counts",
        "Each food class contributes two items to each box",
        "Both lids are fully closed",
        "The boxes are stable in their assigned left and right regions",
    ),
    "d16_sweep_blocks_to_bin": (
        "The broom completed a physical handoff",
        "All five blocks entered the storage bin without direct hand pickup",
        "The broom and dustpan were returned to their marked regions",
        "The storage bin remains upright and stable",
    ),
    "d17_store_laptop_headphones": (
        "The headphones are stably hung on the stand",
        "The laptop is closed to at most five degrees",
        "The laptop is vertical and inserted to the required depth",
        "The vertical stand remains upright",
    ),
    "d18_pack_eggs": (
        "Four eggs occupy four independent carton slots",
        "No egg exceeded the conservative break-impact threshold",
        "The carton lid is closed",
        "The carton remains inside its target region",
    ),
    "d19_memory_cover_sort": (
        "All three blocks were covered from left to right",
        "Covers were removed in red, green, then blue order",
        "The blocks entered the basket in red, green, then blue order",
        "No incorrect reveal was recorded",
    ),
    "d20_microwave_hot_dog_plating": tuple(_d20.SUCCESS_CONDITION_LABELS),
}


D_LONG_TASK_BODIES = {
    "t42_sort_objects": {
        "d06_cracker_box",
        "d06_sugar_box",
        "d06_tomato_soup_can",
        "d06_tuna_fish_can",
        "d06_bowl",
        "d06_mug",
        "d06_basket_box",
        "d06_basket_can",
        "d06_basket_dish",
        *(f"d06_distractor_{index}" for index in range(12)),
    },
    "d07_organize_office_desk": {
        "d07_keyboard",
        "d07_keyboard_frame",
        "d07_mouse",
        "d07_mouse_pad",
        "d07_figurine",
        "d07_stand",
        "d07_alarm_clock",
        "d07_drawer",
        "d07_misc_0",
        "d07_misc_1",
        "d07_misc_2",
    },
    "d08_toast_bread": {
        "d08_toaster",
        "d08_bread_0",
        "d08_bread_1",
        "d08_bread_basket",
        "d08_plate",
    },
    "t31_prepare_coffee": {
        "d09_cabinet",
        "d09_coffee_machine",
        "d09_mug",
        "d09_tray",
    },
    "d10_load_dishwasher": {
        "d10_dishwasher",
        "d10_mug",
        "d10_bowl",
        "d10_plate",
    },
    "d11_presoak_pan": {
        "d11_sink",
        "d11_faucet",
        "d11_pan",
        "d11_sponge",
    },
    "d12_prepare_cutting_station": {
        "d12_drawer",
        "d12_knife",
        "d12_cutting_board",
        "d12_plate",
        "d12_meat",
        "d12_board_goal",
    },
    "d13_prepare_tea_tray": {
        "d13_tray", "d13_kettle", "d13_mug", "d13_tea_bag",
        "d13_cabinet", "d13_service_zone",
    },
    "d14_wash_fruit": {
        "d14_colander", "d14_sink", "d14_faucet", "d14_drying_area",
        "d14_banana", "d14_apple", "d14_orange",
    },
    "d15_pack_matching_lunches": {
        "d15_box_left", "d15_box_right", "d15_lid_left", "d15_lid_right",
        "d15_goal_left", "d15_goal_right",
        *(f"d15_food_{kind}_{index}" for kind in range(3) for index in range(4)),
    },
    "d16_sweep_blocks_to_bin": {
        "d16_broom", "d16_dustpan", "d16_bin", "d16_broom_goal",
        "d16_dustpan_goal", *(f"d16_block_{index}" for index in range(5)),
    },
    "d17_store_laptop_headphones": {
        "d17_headphones", "d17_headphone_stand", "d17_laptop",
        "d17_flat_stand", "d17_vertical_stand",
    },
    "d18_pack_eggs": {
        "d18_carton", "d18_basket", "d18_carton_goal",
        *(f"d18_egg_{index}" for index in range(4)),
    },
    "d19_memory_cover_sort": {
        "d19_red_block", "d19_green_block", "d19_blue_block",
        "d19_cover_0", "d19_cover_1", "d19_cover_2", "d19_basket",
    },
    "d20_microwave_hot_dog_plating": set(_d20.TASK_BODIES),
}

for _index, _name in enumerate(_d20_stages.STAGE_TASKS):
    D_LONG_TASK_BODIES[_name] = set(_d20.TASK_BODIES)
    D_LONG_SUCCESS_CONDITION_LABELS[_name] = _d20_stages.CONDITIONS[_index]

_D_LONG_PREFIXES = tuple(f"d{number:02d}_" for number in range(6, 21))


D_LONG_TASK_BODIES[_d08_variant.TASK_NAME] = set(D_LONG_TASK_BODIES[_d08_variant.BASE_TASK_NAME])
D_LONG_SUCCESS_CONDITION_LABELS[_d08_variant.TASK_NAME] = tuple(
    label.replace("front button", "side lever")
    for label in D_LONG_SUCCESS_CONDITION_LABELS[_d08_variant.BASE_TASK_NAME]
)


def apply_d_long_task_filter(root: ET.Element, task_name: str) -> int:
    """Remove inactive D06--D19 bodies and stale contact pairs.

    The shared source scene may contain task-specific ``contact/exclude``
    entries.  Once an inactive top-level body is removed, MuJoCo rejects an
    otherwise valid filtered scene if an exclude still names that body.  Keep
    only pairs whose two body names survive the task filter.
    """

    active = D_LONG_TASK_BODIES.get(task_name)
    if active is None:
        return 0
    removed = 0
    for worldbody in root.findall("worldbody"):
        for body in list(worldbody.findall("body")):
            name = body.attrib.get("name", "")
            if not name.startswith(_D_LONG_PREFIXES):
                continue
            if name in active:
                continue
            worldbody.remove(body)
            removed += 1

    # Contact excludes are resolved by name during MJCF compilation, so they
    # must be filtered in lockstep with the worldbody.  Include nested bodies
    # (articulation links and free-joint payloads) in the surviving-name set.
    surviving_bodies = {
        body.attrib.get("name", "")
        for body in root.findall(".//body")
        if body.attrib.get("name")
    }
    for contact in root.findall("contact"):
        for exclude in list(contact.findall("exclude")):
            body1 = exclude.attrib.get("body1", "")
            body2 = exclude.attrib.get("body2", "")
            if body1 not in surviving_bodies or body2 not in surviving_bodies:
                contact.remove(exclude)
                removed += 1

    if task_name == "d10_load_dishwasher":
        # Imported bowl/plate shells start at or below the tabletop and
        # produce prolonged GPU contact-correction rocking.  Give loose
        # dishes a small construction gap; gravity still settles them onto
        # the support as dynamic, graspable bodies.
        safe_z = {"d10_mug": 0.950, "d10_bowl": 0.950, "d10_plate": 0.950}
        for body in root.findall("worldbody/body"):
            z = safe_z.get(body.attrib.get("name", ""))
            if z is None:
                continue
            position = body.attrib.get("pos", "0 0 0").split()
            if len(position) == 3:
                position[2] = f"{z:.6f}"
                body.attrib["pos"] = " ".join(position)
    elif task_name == "d12_prepare_cutting_station":
        # The thin knife is authored flush with the drawer tray.  Random yaw
        # lets the GPU solver flip it during reset; raise it so the dynamic
        # knife can settle without changing its XY layout or yaw randomization.
        for body in root.findall("worldbody/body"):
            if body.attrib.get("name") != "d12_knife":
                continue
            position = body.attrib.get("pos", "0 0 0").split()
            if len(position) == 3:
                position[2] = "1.035000"
                body.attrib["pos"] = " ".join(position)
    elif task_name == "d15_pack_matching_lunches":
        # The authored D15 payloads were 22--30 mm below the tabletop after
        # MJCF import (the table top is z=.92).  That deep initial penetration
        # is tolerated by CPU PhysX but trips Isaac's GPU narrowphase during
        # the first reset.  Lift each free body just above the support plane;
        # gravity then settles it onto the box/table without changing the
        # task's XY layout or interaction semantics.
        safe_z = {
            "d15_box_left": 0.903,
            "d15_box_right": 0.903,
            "d15_lid_left": 0.904,
            "d15_lid_right": 0.904,
            **{
                f"d15_food_{kind}_{index}": 0.918
                for kind in range(3)
                for index in range(4)
            },
        }
        for body in root.findall("worldbody/body"):
            name = body.attrib.get("name", "")
            z = safe_z.get(name)
            if z is None:
                continue
            position = body.attrib.get("pos", "0 0 0").split()
            if len(position) != 3:
                continue
            position[2] = f"{z:.6f}"
            body.attrib["pos"] = " ".join(position)
    elif task_name == "d13_prepare_tea_tray":
        # D13's tray and kettle were authored with their collision hulls
        # deeply intersecting the tabletop.  CPU PhysX resolves this on the
        # first frame, while the GPU narrowphase can fail with CUDA 700.  Give
        # the payloads a 1 mm construction gap and let gravity settle them.
        safe_z = {
            "d13_tray": 0.900,
            "d13_kettle": 0.900,
            # The 12 mm-thick bag rests with its center at the 0.896 m
            # tabletop support plane after import; authoring it higher causes
            # a 100+ mm drop and invalidates the reset origin.
            "d13_tea_bag": 0.896,
        }
        for body in root.findall("worldbody/body"):
            name = body.attrib.get("name", "")
            z = safe_z.get(name)
            if z is None:
                continue
            position = body.attrib.get("pos", "0 0 0").split()
            if len(position) != 3:
                continue
            position[2] = f"{z:.6f}"
            body.attrib["pos"] = " ".join(position)
    elif task_name == "d14_wash_fruit":
        # The imported produce shells sit well below their visual meshes;
        # the banana drops about 35 mm during GPU contact resolution.  Lift
        # the loose produce above the tabletop so reset starts without deep
        # penetration or a large correction impulse.
        safe_z = {
            "d14_colander": 0.935,
            "d14_banana": 0.990,
            "d14_apple": 0.990,
            "d14_orange": 0.990,
        }
        for body in root.findall("worldbody/body"):
            z = safe_z.get(body.attrib.get("name", ""))
            if z is None:
                continue
            position = body.attrib.get("pos", "0 0 0").split()
            if len(position) == 3:
                position[2] = f"{z:.6f}"
                body.attrib["pos"] = " ".join(position)
    elif task_name == "d18_pack_eggs":
        # The carton base is authored 22 mm into the tabletop and its lid
        # hinge consequently opens while the carton is corrected by GPU
        # contact.  Lift the carton and loose eggs together before gravity
        # settle so the reset hinge remains at its authored closed angle.
        safe_z = {
            "d18_carton": 1.000,
            **{f"d18_egg_{index}": 1.020 for index in range(4)},
        }
        for body in root.findall("worldbody/body"):
            z = safe_z.get(body.attrib.get("name", ""))
            if z is None:
                continue
            position = body.attrib.get("pos", "0 0 0").split()
            if len(position) == 3:
                position[2] = f"{z:.6f}"
                body.attrib["pos"] = " ".join(position)
    elif task_name == "d16_sweep_blocks_to_bin":
        # The authored broom straddles the table edge and falls to the floor
        # during reset.  Keep it on the tabletop while preserving a loose,
        # graspable tool and the task's block layout.
        for body in root.findall("worldbody/body"):
            if body.attrib.get("name") == "d16_broom":
                body.attrib["pos"] = "-.34 .34 1.000"
            elif body.attrib.get("name") == "d16_dustpan":
                body.attrib["pos"] = "-.18 -.42 .950"
    elif task_name == "d17_store_laptop_headphones":
        # The laptop base is authored coplanar with the table and the
        # headphone earcups sit inside its support shell.  Start both with a
        # clearance gap so GPU contact correction cannot rock them at origin.
        safe_z = {"d17_laptop": 0.980, "d17_headphones": 1.000}
        for body in root.findall("worldbody/body"):
            z = safe_z.get(body.attrib.get("name", ""))
            if z is None:
                continue
            position = body.attrib.get("pos", "0 0 0").split()
            if len(position) == 3:
                position[2] = f"{z:.6f}"
                body.attrib["pos"] = " ".join(position)
    if task_name == _d08_variant.TASK_NAME:
        _d08_variant.configure_source(root)
    return removed


__all__ = [
    "D_LONG_SCENE_XML",
    "D_LONG_SUCCESS_CONDITION_LABELS",
    "D_LONG_TASK_BODIES",
    "D_LONG_TASK_NAMES",
    "D_LONG_TASK_SCENES",
    "apply_d_long_task_filter",
]
