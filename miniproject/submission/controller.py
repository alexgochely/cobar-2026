import numpy as np
from miniproject.simulation import MiniprojectSimulation


class Controller:
    def __init__(self, sim: MiniprojectSimulation):
        from flygym.examples.locomotion import TurningController
        self.turning_controller = TurningController(sim.timestep)

        # ── Olfaction ─────────────────────────────────────────────────
        self.attractive_gain = -1000.0
        self.aversive_gain   = 80.0

        # ── Vision (raw RGB) ──────────────────────────────────────────
        self.roi_top    = 0.0     # tout le ciel inclus
        self.roi_bottom = 0.60    # 10% sous l'horizon

        # Seuil global (réponse proportionnelle au-dessus)
        self.slow_thr = 0.31
        self.slow_min = 0.4

        # Zone de danger central (droit devant)
        self.danger_zone_width = 0.20    # 20% bord interne de chaque œil
        self.danger_thr        = 0.37    # 37% rempli → urgence
        self.danger_speed      = 0.05    # quasi stop
        self.danger_turn       = 1.3     # pivot fort

        # Virage
        self.visual_gain      = 6.0
        self.center_dead_zone = 0.02

        # ── Vent ──────────────────────────────────────────────────────
        self.wind_gain      = 1.0
        self.wind_threshold = 0.5

        # ── Cadence décision ──────────────────────────────────────────
        self.decision_interval    = 0.05
        self._last_decision_time  = -np.inf
        self._last_control_signal = np.array([1.0, 1.0])

        # ── Stop banane ───────────────────────────────────────────────
        self.stop_distance = 2.0
        self.target_xy = np.array(sim.world.banana_xy)

        # ── Debug ─────────────────────────────────────────────────────
        self._last_area_L   = 0.0
        self._last_area_R   = 0.0
        self._last_danger_L = 0.0
        self._last_danger_R = 0.0
        self._last_mode     = "olfaction"

        self.fly = sim.fly

    # ══════════════════════════════════════════════════════════════════
    # MODULES SENSORIELS
    # ══════════════════════════════════════════════════════════════════

    def _detect_grass(self, raw_vision):
        """
        Retourne (area_L, area_R, danger_L, danger_R) :
          - area_*  : fraction sur tout le ROI
          - danger_L : zone droite de l'œil gauche (centre du champ visuel)
          - danger_R : zone gauche de l'œil droit (centre du champ visuel)
        """
        areas = []
        dangers = []
        for i, img in enumerate(raw_vision):
            H, W, _ = img.shape
            roi = img[int(H * self.roi_top):int(H * self.roi_bottom), :, :]
            r = roi[..., 0].astype(float)
            g = roi[..., 1].astype(float)
            b = roi[..., 2].astype(float)
            grass = (g > 80) & (g > 1.5 * r) & (g > 1.5 * b)
            areas.append(grass.mean())

            _, Wr = grass.shape
            zone_w = int(Wr * self.danger_zone_width)
            if i == 0:
                # Œil gauche → zone critique = bord DROIT
                dangers.append(grass[:, -zone_w:].mean())
            else:
                # Œil droit → zone critique = bord GAUCHE
                dangers.append(grass[:, :zone_w].mean())
        return areas[0], areas[1], dangers[0], dangers[1]

    def _visual_analysis(self, raw_vision, current_time):
        area_L, area_R, danger_L, danger_R = self._detect_grass(raw_vision)
        self._last_area_L, self._last_area_R = area_L, area_R
        self._last_danger_L, self._last_danger_R = danger_L, danger_R

        # ── DANGER : obstacle imminent droit devant ──
        danger_max = max(danger_L, danger_R)
        if danger_max > self.danger_thr:
            # Pivot vers le côté le plus dégagé (basé sur l'aire globale)
            if area_L > area_R:
                # Plus d'herbe à gauche → pivote à droite
                drive = np.array([self.danger_turn, self.danger_speed])
            else:
                # Plus d'herbe à droite → pivote à gauche
                drive = np.array([self.danger_speed, self.danger_turn])
            return "danger", drive

        # ── Comportement proportionnel normal ──
        area_max = max(area_L, area_R)
        if area_max < self.slow_thr:
            return "clear", None

        intensity = np.clip(
            (area_max - self.slow_thr) / (1.0 - self.slow_thr),
            0.0, 1.0
        )

        diff = area_L - area_R
        speed = 1.0 - intensity * (1.0 - self.slow_min)
        turn_strength = min(self.visual_gain * intensity * abs(diff), 1.5)

        if abs(diff) < self.center_dead_zone:
            drive = np.array([
                speed + 0.3 * intensity, speed * (1.0 - 0.5 * intensity)
            ])
        elif diff > 0:
            # Plus d'herbe à gauche → vire à droite
            drive = np.array([
                speed + 0.2 * intensity,
                max(speed - turn_strength, 0.05),
            ])
        else:
            # Plus d'herbe à droite → vire à gauche
            drive = np.array([
                max(speed - turn_strength, 0.05),
                speed + 0.2 * intensity,
            ])

        return "turn", drive

    def _olfactory_drive(self, odor_intensities):
        attractive = np.average(
            odor_intensities[:, 0].reshape(2, 2), axis=0, weights=[9, 1]
        )
        attractive_bias = 0.0
        if attractive.mean() > 0:
            attractive_bias = self.attractive_gain * (
                (attractive[0] - attractive[1]) / attractive.mean()
            )
        aversive_bias = 0.0
        if odor_intensities.shape[1] > 1:
            aversive = np.average(
                odor_intensities[:, 1].reshape(2, 2), axis=0, weights=[10, 0]
            )
            if aversive.mean() > 0:
                aversive_bias = self.aversive_gain * (
                    (aversive[0] - aversive[1]) / aversive.mean()
                )
        bias = attractive_bias + aversive_bias
        bias_norm = np.tanh(bias ** 2) * np.sign(bias)
        drive = np.ones(2)
        drive[int(bias_norm > 0)] -= np.abs(bias_norm) * 0.75
        return drive

    def _wind_drive(self, antenna_data):
        if antenna_data is None or not isinstance(antenna_data, dict):
            return np.array([1.0, 1.0]), False
        try:
            left_force  = np.linalg.norm(antenna_data["l"]["qfrc_passive"])
            right_force = np.linalg.norm(antenna_data["r"]["qfrc_passive"])
        except (KeyError, TypeError):
            return np.array([1.0, 1.0]), False
        total = left_force + right_force
        if total < self.wind_threshold:
            return np.array([1.0, 1.0]), False
        diff = left_force - right_force
        bias = np.tanh(self.wind_gain * diff / (total + 1e-6))
        drive = np.ones(2)
        drive[int(bias > 0)] -= np.abs(bias) * 0.75
        return drive, True

    # ══════════════════════════════════════════════════════════════════
    # FUSION
    # ══════════════════════════════════════════════════════════════════

    def _compute_control_signal(self, odor, raw_vision, antenna, current_time):
        drive = self._olfactory_drive(odor)
        self._last_mode = "olfaction"

        wind_drive, wind_detected = self._wind_drive(antenna)
        if wind_detected:
            drive = 0.5 * drive + 0.5 * wind_drive
            self._last_mode = "wind"

        action, payload = self._visual_analysis(raw_vision, current_time)
        if action == "turn":
            odor_strength = float(odor[:, 0].mean())  
            olf_weight = 0.2 + 0.5 * np.tanh(odor_strength * 1e6)
            drive = olf_weight * drive + (1.0 - olf_weight) * payload
            self._last_mode = "vision"
        elif action == "danger":
            drive = payload
            self._last_mode = "danger"

        return np.clip(drive, 0.05, 1.4)

    # ══════════════════════════════════════════════════════════════════
    # STEP
    # ══════════════════════════════════════════════════════════════════

    def step(self, sim):
        fly_pos = sim.mj_data.body(f"{sim.fly.name}/").xpos[:2]
        if np.linalg.norm(fly_pos - self.target_xy) < self.stop_distance:
            return self.turning_controller.step(np.array([0.0, 0.0]))

        t = sim.mj_data.time
        if t - self._last_decision_time >= self.decision_interval:
            odor       = sim.get_olfaction(sim.fly.name)
            raw_vision = sim.get_raw_vision(sim.fly.name)
            antenna    = sim.get_antenna_data(sim.fly.name)
            self._last_control_signal = self._compute_control_signal(
                odor, raw_vision, antenna, t
            )
            self._last_decision_time = t

        return self.turning_controller.step(self._last_control_signal)