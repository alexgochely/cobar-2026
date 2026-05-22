"""Reactive controller for the FlyGym mini-project.

The fly must walk toward a target (banana) guided by smell while reacting to
three hazards detected through other senses:

  * Grass patches on the ground (vision, lower image) -> steer around them.
  * An attacking dragonfly (vision, sky) -> dash straight ahead to break its
    lock-on, then freeze so the now-stationary target is hard to hit.
  * Wind (antenna mechanosensation) -> bias steering against the gust.

Each sense produces a steering "drive" -- a length-2 array of left/right wheel
speeds. The senses are fused with a fixed priority: dragonfly escape overrides
everything, grass avoidance overrides olfaction, and wind blends with
olfaction. Olfaction is the baseline behavior when nothing else fires.
"""

import numpy as np

from miniproject.simulation import MiniprojectSimulation


# Control loop runs at a fixed cadence; one decision every CONTROL_DT seconds.
CONTROL_DT = 0.02


class Controller:
    """Fuses olfaction, vision and wind cues into left/right wheel drives."""

    def __init__(self, sim: MiniprojectSimulation):
        from flygym.examples.locomotion import TurningController

        self.turning_controller = TurningController(sim.timestep)
        self.fly = sim.fly

        # ── Olfaction ────────────────────────────────────────────────────
        # Steering bias gains for the attractive (target) and aversive odors.
        self.attractive_gain = -100.0
        self.aversive_gain = 10.0

        # ── Vision: grass avoidance ───────────────────────────────────────
        # Grass is detected in the lower part of the image (the ground). A
        # detection above the threshold triggers an avoidance turn.
        self.grass_roi_top = 0.0
        self.grass_roi_bottom = 0.4
        self.grass_threshold = 0.035
        self.grass_zone_width = 0.5      # fraction of the eye's width to inspect
        self.grass_outer_speed = 1.0     # fast (outer) wheel during avoidance
        self.grass_inner_speed = 0.05    # slow (inner) wheel during avoidance

        # Hold the avoidance turn for a short time to avoid flickering between
        # "danger" and "clear" on frame-to-frame noise.
        self.grass_hold_max = int(0.3 / CONTROL_DT)
        self.grass_hold = 0

        # ── Vision: dragonfly escape ──────────────────────────────────────
        # The dragonfly is detected by its red color in the sky (the rest of
        # the scene is blue sky and green ground/grass). Its eyes are dark red
        # throughout the approach and its head turns bright red at the dive, so
        # even a small amount of red is a reliable cue.
        #
        # Strategy: once red is confirmed for a few frames, dash straight ahead
        # at full speed. The dragonfly locks its trajectory at a fixed distance
        # and no longer corrects, so building lateral momentum moves us out of
        # its strike line. After the dash we freeze: a stationary target plus
        # the lateral offset makes the strike miss.
        self.dragonfly_roi_top = 0.0
        self.dragonfly_roi_bottom = 0.7
        self.dragonfly_threshold = 5e-6        # a hint of red is enough
        self.dragonfly_confirm_steps = 25      # frames of confirmation before dashing
        self.dragonfly_confirm = 0

        self.dragonfly_dash_max = int(0.10 / CONTROL_DT)
        self.dragonfly_dash = 0
        self.dragonfly_freeze_max = self.dragonfly_dash_max
        self.dragonfly_freeze = 0
        self.escape_speed = 4.0                # full throttle on both wheels

        # ── Wind ──────────────────────────────────────────────────────────
        self.wind_gain = 1.0
        self.wind_threshold = 0.5

        # ── Drive limits ──────────────────────────────────────────────────
        # A higher speed floor applies only while escaping the dragonfly, where
        # lateral momentum matters. Otherwise the floor is low so the fly can
        # pivot sharply (inner wheel close to zero).
        self.escape_floor = 0.1
        self.normal_floor = 0.05
        self.drive_ceiling = 5.0

        # ── Decision cadence and stop condition ───────────────────────────
        self._last_decision_time = -np.inf
        self._last_control_signal = np.array([1.0, 1.0])

        self.stop_distance = 2.0
        self.target_xy = np.array(sim.world.banana_xy)

    # ════════════════════════════════════════════════════════════════════
    # VISION
    # ════════════════════════════════════════════════════════════════════
    def _detect_grass(self, raw_vision):
        """Return the grass fraction seen toward the center by each eye.

        Grass is green: green clearly dominates red and blue and is bright
        enough. Each eye only inspects the inner edge of its field (toward the
        direction of travel), so the returned values approximate the hazard
        directly ahead-left and ahead-right.
        """
        dangers = []
        for eye_index, img in enumerate(raw_vision):
            height = img.shape[0]
            roi = img[int(height * self.grass_roi_top):
                      int(height * self.grass_roi_bottom), :, :]
            r = roi[..., 0].astype(float)
            g = roi[..., 1].astype(float)
            b = roi[..., 2].astype(float)
            brightness = (r + g + b) / 3
            grass = (g > 50) & (g > 1.25 * r) & (g > 1.25 * b) & (brightness > 40)

            zone_w = int(grass.shape[1] * self.grass_zone_width)
            if eye_index == 0:  # left eye -> inspect its right edge (field center)
                dangers.append(grass[:, -zone_w:].mean())
            else:               # right eye -> inspect its left edge
                dangers.append(grass[:, :zone_w].mean())
        return dangers[0], dangers[1]

    def _detect_dragonfly(self, raw_vision):
        """Return the red fraction seen by each eye (dragonfly cue).

        Red is unique to the dragonfly here: its dark-red eyes (~153, 26, 26)
        are visible during the whole approach and its head turns bright red
        (~255, 0, 0) at the dive. The threshold on red is kept low enough to
        catch the eyes, giving an early warning rather than only firing at the
        last moment.
        """
        out = []
        for img in raw_vision:
            height = img.shape[0]
            roi = img[int(height * self.dragonfly_roi_top):
                      int(height * self.dragonfly_roi_bottom), :, :]
            r = roi[..., 0].astype(float)
            g = roi[..., 1].astype(float)
            b = roi[..., 2].astype(float)
            red = (r > 120) & (r > 2.0 * g) & (r > 2.0 * b)
            out.append(red.mean())
        return out[0], out[1]

    def _visual_analysis(self, raw_vision, prefer_left):
        """Run the vision-based behaviors in priority order.

        Returns a (mode, drive) pair. ``drive`` is None when vision yields no
        override and the caller should keep the olfaction/wind drive.
        """
        # ── Dragonfly (highest priority) ──────────────────────────────────
        dragonfly_L, dragonfly_R = self._detect_dragonfly(raw_vision)
        if max(dragonfly_L, dragonfly_R) > self.dragonfly_threshold:
            self.dragonfly_confirm += 1
        else:
            # A single isolated flicker should not reset the count abruptly.
            self.dragonfly_confirm = max(0, self.dragonfly_confirm - 1)

        if self.dragonfly_confirm >= self.dragonfly_confirm_steps:
            self.dragonfly_dash = self.dragonfly_dash_max

        if self.dragonfly_dash > 0:
            self.dragonfly_dash -= 1
            if self.dragonfly_dash == 0:
                self.dragonfly_confirm = 0
                self.dragonfly_freeze = self.dragonfly_freeze_max
            # Dash straight ahead at full speed to build lateral offset.
            return "escape", np.array([self.escape_speed, self.escape_speed])

        # ── Freeze after the dash: stay still so the locked strike misses. ──
        if self.dragonfly_freeze > 0:
            self.dragonfly_freeze -= 1
            return "freeze", np.array([0.0, 0.0])

        # ── Grass avoidance ───────────────────────────────────────────────
        danger_L, danger_R = self._detect_grass(raw_vision)
        if max(danger_L, danger_R) > self.grass_threshold:
            self.grass_hold = self.grass_hold_max
        if self.grass_hold > 0:
            self.grass_hold -= 1
            diff = danger_L - danger_R
            if abs(diff) < 0.05:
                turn_left = prefer_left           # ambiguous -> follow olfaction
            else:
                turn_left = diff < 0              # more grass on the right -> go left
            if turn_left:
                return "danger", np.array([self.grass_inner_speed,
                                           self.grass_outer_speed])
            return "danger", np.array([self.grass_outer_speed,
                                       self.grass_inner_speed])

        return "clear", None


    # ════════════════════════════════════════════════════════════════════
    # OLFACTION
    # ════════════════════════════════════════════════════════════════════
    def _olfactory_drive(self, odor):
        """Baseline steering toward the attractive odor, away from aversive.

        The left/right odor intensities are turned into a steering bias, which
        is squashed and used to slow down the wheel on the side we turn toward.
        """
        attractive = np.average(odor[:, 0].reshape(2, 2), axis=0, weights=[9, 1])
        attractive_bias = 0.0
        if attractive.mean() > 0:
            attractive_bias = self.attractive_gain * (
                (attractive[0] - attractive[1]) / attractive.mean()
            )
        aversive_bias = 0.0
        if odor.shape[1] > 1:
            aversive = np.average(odor[:, 1].reshape(2, 2), axis=0, weights=[10, 0])
            if aversive.mean() > 0:
                aversive_bias = self.aversive_gain * (
                    (aversive[0] - aversive[1]) / aversive.mean()
                )
        bias = attractive_bias + aversive_bias
        bias_norm = np.tanh(bias ** 2) * np.sign(bias)
        drive = np.ones(2)
        drive[int(bias_norm > 0)] -= np.abs(bias_norm) * 0.75
        return drive

    # ════════════════════════════════════════════════════════════════════
    # WIND
    # ════════════════════════════════════════════════════════════════════
    def _wind_drive(self, antenna):
        """Steer against the gust using the passive force on each antenna.

        Returns (drive, detected). ``detected`` is False when there is no
        antenna data or the gust is below threshold, in which case the drive is
        neutral and the caller ignores it.
        """
        if not isinstance(antenna, dict):
            return np.array([1.0, 1.0]), False
        try:
            L = np.linalg.norm(antenna["l"]["qfrc_passive"])
            R = np.linalg.norm(antenna["r"]["qfrc_passive"])
        except (KeyError, TypeError, AttributeError):
            return np.array([1.0, 1.0]), False

        if L + R < self.wind_threshold:
            return np.array([1.0, 1.0]), False

        bias = np.tanh(self.wind_gain * (L - R) / (L + R + 1e-6))
        drive = np.ones(2)
        drive[int(bias > 0)] -= np.abs(bias) * 0.75
        return drive, True

    # ════════════════════════════════════════════════════════════════════
    # FUSION
    # ════════════════════════════════════════════════════════════════════
    def _compute_control_signal(self, odor, raw_vision, antenna):
        """Combine all senses into a single, clipped left/right drive."""
        drive = self._olfactory_drive(odor)

        # Preferred turn side = the side olfaction already wants to turn toward.
        prefer_left = drive[0] < drive[1]

        wind_drive, wind_detected = self._wind_drive(antenna)
        if wind_detected:
            drive = 0.5 * drive + 0.5 * wind_drive

        action, payload = self._visual_analysis(raw_vision, prefer_left)
        if action in ("escape", "freeze", "danger"):
            drive = payload  # vision overrides olfaction/wind

        # Higher speed floor only while dashing (needs lateral momentum);
        # otherwise a low floor lets the fly pivot sharply.
        floor = self.escape_floor if self.dragonfly_dash > 0 else self.normal_floor
        return np.clip(drive, floor, self.drive_ceiling)

    # ════════════════════════════════════════════════════════════════════
    # STEP
    # ════════════════════════════════════════════════════════════════════
    def step(self, sim):
        """Advance one simulation step, recomputing the drive on cadence."""
        pos = sim.mj_data.body(f"{sim.fly.name}/").xpos[:2]
        if np.linalg.norm(pos - self.target_xy) < self.stop_distance:
            return self.turning_controller.step(np.array([0.0, 0.0]))

        t = sim.mj_data.time
        if t - self._last_decision_time >= CONTROL_DT:
            odor = sim.get_olfaction(sim.fly.name)
            vision = sim.get_raw_vision(sim.fly.name)
            antenna = sim.get_antenna_data(sim.fly.name)
            self._last_control_signal = self._compute_control_signal(
                odor, vision, antenna
            )
            self._last_decision_time = t

        return self.turning_controller.step(self._last_control_signal)
