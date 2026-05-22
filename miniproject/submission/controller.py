import numpy as np
from miniproject.simulation import MiniprojectSimulation


class Controller:

    def __init__(self, sim: MiniprojectSimulation):
        from flygym.examples.locomotion import TurningController
        self.turning_controller = TurningController(sim.timestep)

        # ── Olfaction ─────────────────────────────────────────
        self.attractive_gain = -1000.0
        self.aversive_gain = 10.0
        
        
        # ── Vision (ROI) ──────────────────────────────────────
        self.roi_top = 0.0
        self.roi_bottom = 0.4

        # Seuil danger frontal (déclenche l'évitement)
        self.danger_thr = 0.035

        # Zone danger frontale (bord interne de chaque œil)
        self.danger_zone_width = 0.5
        self.danger_speed = 1  # roue rapide pendant l'évitement
        self.danger_turn = 0.05    # roue lente (côté du virage)

        # ── Refractory danger (anti-flicker override) ────────
        self.danger_hold_sec = 0.3
        self.danger_hold_max = int(self.danger_hold_sec / 0.02)  # vs decision_interval
        self.danger_hold = 0

        # ── Lissage commande ─────────────────────────────────
        self.drive_smoothing = 0.0
        self.previous_drive = np.array([1.0, 1.0])

        # ── Vent ─────────────────────────────────────────────
        self.wind_gain = 1.0
        self.wind_threshold = 0.5

        # ── Détection dragonfly ─────────────────────
        self.dragonfly_roi_top = 0.0
        self.dragonfly_roi_bottom = 0.35    # dans le ciel
        self.dragonfly_thr = 0.005       
        self.dragonfly_hold_sec = 1.0       # durée approxi de l'attaque
        self.dragonfly_hold_max = int(self.dragonfly_hold_sec / 0.02)
        self.dragonfly_hold = 0
        self.escape_speed = 1.2             
        self.escape_turn = 0.0              

        # ── Cadence ──────────────────────────────────────────
        self.decision_interval = 0.02
        self._last_decision_time = -np.inf
        self._last_control_signal = np.array([1.0, 1.0])

        # ── Stop ─────────────────────────────────────────────
        self.stop_distance = 2.0
        self.target_xy = np.array(sim.world.banana_xy)

        # ── Debug ────────────────────────────────────────────
        self._last_danger_L = 0.0
        self._last_danger_R = 0.0
        self._last_mode = "olfaction"

        self.fly = sim.fly

    # ════════════════════════════════════════════════════════════
    # VISION
    # ════════════════════════════════════════════════════════════
    def _detect_grass(self, raw_vision):
        dangers = []
        for i, img in enumerate(raw_vision):
            H, W, _ = img.shape
            roi = img[int(H * self.roi_top):int(H * self.roi_bottom), :, :]
            r = roi[..., 0].astype(float)
            g = roi[..., 1].astype(float)
            b = roi[..., 2].astype(float)
            brightness = (r + g + b) / 3
            grass = (
                (g > 50)
                & (g > 1.25 * r)
                & (g > 1.25 * b)
                & (brightness > 40)
            )

            _, Wr = grass.shape
            zone_w = int(Wr * self.danger_zone_width)
            if i == 0:  # œil gauche → bord droit (centre du champ)
                dangers.append(grass[:, -zone_w:].mean())
            else:       # œil droit → bord gauche
                dangers.append(grass[:, :zone_w].mean())

        return dangers[0], dangers[1]
    
    def _detect_dragonfly(self, raw_vision):
        
        dragonfly = []
        for i, img in enumerate(raw_vision):
            H, W, _ = img.shape
            roi = img[int(H * self.dragonfly_roi_top):int(H * self.dragonfly_roi_bottom), :, :]
            r = roi[..., 0].astype(float)
            g = roi[..., 1].astype(float)
            b = roi[..., 2].astype(float)
            red = (r > 100) & (r > 1.5 * g) & (r > 1.5 * b)
            dragonfly.append(red.mean())
        return dragonfly[0], dragonfly[1]

    def _visual_analysis(self, raw_vision, pref_left):

        #Détection dragonfly (prioritaire)
        dragonfly_L, dragonfly_R = self._detect_dragonfly(raw_vision)
        dragonfly_max = max(dragonfly_L, dragonfly_R)
        if dragonfly_max > self.dragonfly_thr:
            self.dragonfly_hold = self.dragonfly_hold_max
        if self.dragonfly_hold > 0:
            self.dragonfly_hold -= 1
            # Fuir du côté OPPOSÉ à la dragonfly
            if dragonfly_L > dragonfly_R:
                drive = np.array([self.escape_speed, self.escape_turn])
            else:
                drive = np.array([self.escape_turn, self.escape_speed])
            return "escape", drive
        
        danger_L, danger_R = self._detect_grass(raw_vision)
        self._last_danger_L, self._last_danger_R = danger_L, danger_R

        danger_max = max(danger_L, danger_R)
        if danger_max > self.danger_thr:
            self.danger_hold = self.danger_hold_max
        if self.danger_hold > 0:
            self.danger_hold -= 1
            diff = danger_L - danger_R
            if abs(diff) < 0.05:
                # quasi symétrique → suivre l'olfaction
                turn_left = pref_left
            else:
                # sinon fuir le côté le plus encombré
                turn_left = diff < 0   # plus d'herbe à droite → tourner à gauche
            if turn_left:
                drive = np.array([self.danger_turn, self.danger_speed])
            else:
                drive = np.array([self.danger_speed, self.danger_turn])
            return "danger", drive

        return "clear", None

    # ════════════════════════════════════════════════════════════
    # OLFACTION
    # ════════════════════════════════════════════════════════════
    def _olfactory_drive(self, odor):
        attractive = np.average(
            odor[:, 0].reshape(2, 2), axis=0, weights=[9, 1]
        )
        attractive_bias = 0.0
        if attractive.mean() > 0:
            attractive_bias = self.attractive_gain * (
                (attractive[0] - attractive[1]) / attractive.mean()
            )

        aversive_bias = 0.0
        if odor.shape[1] > 1:
            aversive = np.average(
                odor[:, 1].reshape(2, 2), axis=0, weights=[10, 0]
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

    # ════════════════════════════════════════════════════════════
    # VENT
    # ════════════════════════════════════════════════════════════
    def _wind_drive(self, antenna):
        if antenna is None or not isinstance(antenna, dict):
            return np.array([1.0, 1.0]), False
        try:
            L = np.linalg.norm(antenna["l"]["qfrc_passive"])
            R = np.linalg.norm(antenna["r"]["qfrc_passive"])
        except (KeyError, TypeError, AttributeError):
            return np.array([1.0, 1.0]), False

        total = L + R
        if total < self.wind_threshold:
            return np.array([1.0, 1.0]), False

        bias = np.tanh(self.wind_gain * (L - R) / (total + 1e-6))
        drive = np.ones(2)
        drive[int(bias > 0)] -= np.abs(bias) * 0.75
        return drive, True

    # ════════════════════════════════════════════════════════════
    # FUSION
    # ════════════════════════════════════════════════════════════
    def _compute_control_signal(self, odor, raw_vision, antenna):
        drive = self._olfactory_drive(odor)
        attractive = odor[:,0].reshape(2,2).mean(axis=0)
        self._last_mode = "olfaction"

        # côté préféré = côté vers lequel l'olfaction veut tourner
        pref_left = drive[0] < drive[1]

        wind_drive, wind_detected = self._wind_drive(antenna)
        if wind_detected:
            drive = 0.5 * drive + 0.5 * wind_drive
            self._last_mode = "wind"

        action, payload = self._visual_analysis(raw_vision, pref_left)

        if action == "escape":
            drive = payload
            self._last_mode = "escape"
            # DEBUG TEMPORAIRE — à supprimer pour le rendu final
            if not hasattr(self, '_escape_count'):
                self._escape_count = 0
            self._escape_count += 1
            if self._escape_count % 10 == 1:
                print(f"⚠️  ESCAPE déclenché ! (occurrence #{self._escape_count})")
        elif action == "danger":
            drive = payload   # override : évitement herbe
            self._last_mode = "danger"

        drive = np.clip(drive, 0.05, 1.4)
        drive = (
            self.drive_smoothing * self.previous_drive
            + (1.0 - self.drive_smoothing) * drive
        )
        self.previous_drive = drive.copy()
        return drive

    # ════════════════════════════════════════════════════════════
    # STEP
    # ════════════════════════════════════════════════════════════
    def step(self, sim):
        pos = sim.mj_data.body(f"{sim.fly.name}/").xpos[:2]
        if np.linalg.norm(pos - self.target_xy) < self.stop_distance:
            return self.turning_controller.step(np.array([0.0, 0.0]))

        t = sim.mj_data.time
        if t - self._last_decision_time >= self.decision_interval:
            odor = sim.get_olfaction(sim.fly.name)
            vision = sim.get_raw_vision(sim.fly.name)
            antenna = sim.get_antenna_data(sim.fly.name)
            self._last_control_signal = self._compute_control_signal(
                odor, vision, antenna
            )
            self._last_decision_time = t

        return self.turning_controller.step(self._last_control_signal)
    
    
    
