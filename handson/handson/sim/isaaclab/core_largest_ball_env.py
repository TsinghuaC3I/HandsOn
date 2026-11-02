"""Paper reset variants for Core t27; shared geometry and success rules.

The hand-only cloud was used by all eight multitask models and 66/100 DP
trials. The remaining 34 DP trials used the later full-arm cloud. Selecting
a cloud alone does not certify equivalence of every historical dependency.
See evaluation/core_largest_ball_provenance.json.
"""

from .abc_collection_env import HandsOnABCIsaacLabEnv


class CoreLargestBallEnv(HandsOnABCIsaacLabEnv):
    def __init__(
        self, task_name="t27_place_largest_fitting_ball", *,
        reset_clearance_scope="hand_bodies", **kwargs,
    ):
        if task_name != "t27_place_largest_fitting_ball":
            raise ValueError("CoreLargestBallEnv supports only t27")
        if reset_clearance_scope not in ("hand_bodies", "arm_bodies"):
            raise ValueError("reset_clearance_scope must be hand_bodies or arm_bodies")
        self.reset_clearance_scope = reset_clearance_scope
        super().__init__(task_name, **kwargs)

    def _ycb02_reset_clearance_points(self):
        if self.reset_clearance_scope == "hand_bodies":
            return (*self._a06_hand_points("right"), *self._a06_hand_points("left"))
        return super()._ycb02_reset_clearance_points()

    def _ycb02_reset_xy_positions(self):
        layout = super()._ycb02_reset_xy_positions()
        self._ycb02_reset_layout_diagnostics["clearance_scope"] = self.reset_clearance_scope
        return layout
