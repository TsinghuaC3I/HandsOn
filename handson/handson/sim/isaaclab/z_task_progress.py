"""Physical event history for the provisional Z laboratory operations.

Pipette volume is a plunger-stroke estimate, not a fluid solver.
"""
from dataclasses import dataclass, asdict
import math


@dataclass
class ZTaskProgress:
    previous_stroke: float = 0.0
    previous_angle: float = 0.0
    angle_initialized: bool = False
    turn: float = 0.0
    turn_completed: bool = False
    lifted: bool = False
    aspirated_ul: float = 0.0
    dispensed_ul: float = 0.0
    spilled_ul: float = 0.0
    held: bool = False
    stable_seconds: float = 0.0
    succeeded: bool = False

    def pipette(self, *, stroke, source, destination, hand_contact, on_stand,
                released, quiet, dt, capacity_ul, hold_seconds):
        stroke = min(1., max(0., float(stroke)))
        change = stroke - self.previous_stroke
        self.previous_stroke = stroke
        self.held = self.held or hand_contact
        if self.held:
            if change < 0 and source and hand_contact:
                self.aspirated_ul = min(capacity_ul, self.aspirated_ul-change*capacity_ul)
            if change > 0 and self.aspirated_ul > 0:
                amount = min(self.aspirated_ul, change*capacity_ul)
                self.aspirated_ul -= amount
                if destination and hand_contact:
                    self.dispensed_ul += amount
                else:
                    self.spilled_ul += amount
        complete = self.dispensed_ul >= .9*capacity_ul and self.spilled_ul <= .05*capacity_ul
        self._hold(complete and on_stand and released and quiet, dt, hold_seconds)

    def cap(self, *, angle, seated, axial_lift, role_contacts, released,
            in_dish, quiet, dt, required_turn, lift_distance, hold_seconds):
        if self.angle_initialized:
            change = math.atan2(math.sin(angle-self.previous_angle),math.cos(angle-self.previous_angle))
            if seated and role_contacts and abs(change) < .25:
                self.turn = max(0., self.turn+change)
        self.previous_angle = angle
        self.angle_initialized = True
        if self.turn >= required_turn:
            self.turn_completed = True
        if self.turn_completed and axial_lift >= lift_distance:
            self.lifted = True
        self._hold(self.lifted and in_dish and released and quiet, dt, hold_seconds)

    def _hold(self, valid, dt, seconds):
        self.stable_seconds = self.stable_seconds + dt if valid else 0.
        self.succeeded = self.succeeded or self.stable_seconds >= seconds

    def values(self):
        return [float(value) for value in asdict(self).values()]

    @classmethod
    def from_values(cls, values):
        """Decode the recorded event history without advancing success logic."""
        from dataclasses import fields
        values = list(values)
        names = fields(cls)
        if len(values) != len(names) or not all(math.isfinite(float(v)) for v in values):
            raise ValueError('Z task history must contain 12 finite values')
        decoded = {}
        for field, value in zip(names, values):
            if field.type is bool:
                if value not in (0, 1):
                    raise ValueError('Z history boolean must be 0 or 1: '+field.name)
                decoded[field.name] = bool(value)
            else:
                decoded[field.name] = float(value)
        if any(decoded[name] < 0 for name in ('turn','aspirated_ul','dispensed_ul','spilled_ul','stable_seconds')):
            raise ValueError('Z history quantities cannot be negative')
        return cls(**decoded)
