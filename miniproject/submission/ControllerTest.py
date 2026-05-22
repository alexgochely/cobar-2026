import numpy as np
from miniproject.simulation import MiniprojectSimulation


class Controller:

    def __init__(self, sim: MiniprojectSimulation):
        from flygym.examples.locomotion import TurningController
        self.turning_controller = TurningController(sim.timestep)

        # ── Olfaction ─────────────────────────────────────────
        self.attractive_gain = -100.0
        self.aversive_gain = 10.0

        # ── Vision: grass (ROI = lower image, ground) ─────────
        self.roi_top = 0.0
        self.roi_bottom = 0.4
        self.danger_thr = 0.035
        self.danger_zone_width = 0.5
        self.danger_speed = 1.0     # roue rapide pendant l'évitement
        self.danger_turn = 0.05     # roue lente (côté du virage)

        # ── Refractory danger (anti-flicker override) ────────
        self.danger_hold_sec = 0.3
        self.danger_hold_max = int(self.danger_hold_sec / 0.02)
        self.danger_hold = 0

        # ── Vent ─────────────────────────────────────────────
        self.wind_gain = 1.0
        self.wind_threshold = 0.5

        # ── Détection dragonfly (ROI = ciel, haut de l'image) ──
        # On détecte le ROUGE : les yeux (rouge sombre) sont visibles
        # pendant toute l'approche, et la tête devient rouge pur au
        # plongeon (dist<=15). Le rouge est unique (ciel bleu, sol/herbe
        # verts), donc même un tout petit peu suffit. Dès qu'on voit du
        # rouge on COMPTE quelques pas de confirmation, puis on DASH.
        self.dragonfly_roi_top = 0.0
        self.dragonfly_roi_bottom = 0.7
        self.dragonfly_thr = 0.000005          # un soupçon de rouge suffit (yeux)
        self.dragonfly_confirm_steps = 25    # pas de confirmation avant le dash
        self.dragonfly_confirm = 0
        self.dragonfly_hold_sec = 0.10       # durée du dash
        self.dragonfly_hold_max = int(self.dragonfly_hold_sec / 0.02)
        self.dragonfly_hold = 0

        self.dragonfly_freeze_max = self.dragonfly_hold_max
        self.dragonfly_freeze = 0

        # Stratégie d'esquive : foncer tout droit, ne jamais s'arrêter.
        # La libellule verrouille sa trajectoire à dist=15 et ne corrige
        # plus ; toute vitesse latérale au moment du verrou nous sort de
        # la ligne de frappe.
        self.escape_speed = 4           # plein gaz, deux roues

        # ── Lissage commande ─────────────────────────────────
        self.drive_smoothing = 0.0
        self.previous_drive = np.array([1.0, 1.0])

        # ── Plancher de vitesse ──────────────────────────────
        # Plancher haut UNIQUEMENT pendant la menace libellule (élan
        # latéral). Sinon plancher bas pour pivoter franchement.
        self.drive_floor = 0.1
        self.drive_ceiling = 5

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
        self._escape_count = 0

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
        # ROUGE dans le ciel. Les yeux sont rouge sombre ~(153,26,26),
        # la tête au plongeon est rouge pur ~(255,0,0). On baisse le
        # seuil sur r pour attraper aussi les yeux (donc détection
        # précoce sur toute l'approche, pas seulement au plongeon).
        out = []
        for img in raw_vision:
            H, W, _ = img.shape
            roi = img[
                int(H * self.dragonfly_roi_top):int(H * self.dragonfly_roi_bottom),
                :, :,
            ]
            r = roi[..., 0].astype(float)
            g = roi[..., 1].astype(float)
            b = roi[..., 2].astype(float)
            # rouge dominant : r modéré-haut, g et b nettement plus bas
            red = (r > 120) & (r > 2.0 * g) & (r > 2.0 * b)
            out.append(red.mean())
        return out[0], out[1]

    def _visual_analysis(self, raw_vision, pref_left):
        # ── Dragonfly (prioritaire) ──────────────────────────
        # Dès qu'on voit un peu de rouge, on compte quelques pas de
        # confirmation, puis on engage un dash sur une durée fixe.
        dragonfly_L, dragonfly_R = self._detect_dragonfly(raw_vision)
        dragonfly_max = max(dragonfly_L, dragonfly_R)

        if dragonfly_max > self.dragonfly_thr:
            self.dragonfly_confirm += 1
        else:
            # un flicker isolé ne réinitialise pas brutalement
            self.dragonfly_confirm = max(0, self.dragonfly_confirm - 1)

        # confirmation atteinte → on engage le dash pour une durée fixe
        if self.dragonfly_confirm >= self.dragonfly_confirm_steps:
            self.dragonfly_hold = self.dragonfly_hold_max

        if self.dragonfly_hold > 0:
            self.dragonfly_hold -= 1
            if self.dragonfly_hold == 0:
                self.dragonfly_confirm = 0           # reset après le dash
                self.dragonfly_freeze = self.dragonfly_freeze_max  # puis on fige
            # Dash par l'élan : on fonce TOUT DROIT, plein gaz.
            drive = np.array([self.escape_speed, self.escape_speed])
            return "escape", drive
 
        # ── Freeze post-dash ─────────────────────────────────
        # Après le dash, on se fige complètement pendant la même durée :
        # cible immobile, le dash latéral nous a déjà sortis de la ligne.
        if self.dragonfly_freeze > 0:
            self.dragonfly_freeze -= 1
            return "freeze", np.array([0.0, 0.0])

        # ── Herbe ────────────────────────────────────────────
        danger_L, danger_R = self._detect_grass(raw_vision)
        self._last_danger_L, self._last_danger_R = danger_L, danger_R

        danger_max = max(danger_L, danger_R)
        if danger_max > self.danger_thr:
            self.danger_hold = self.danger_hold_max
        if self.danger_hold > 0:
            self.danger_hold -= 1
            diff = danger_L - danger_R
            if abs(diff) < 0.05:
                turn_left = pref_left
            else:
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
            self._escape_count += 1
            if self._escape_count % 10 == 1:
                print(f"⚠️  ESCAPE (dash tout droit) — occurrence #{self._escape_count}")
        elif action == "danger":
            drive = payload   # override : évitement herbe
            self._last_mode = "danger"

        # Plancher de vitesse : haut UNIQUEMENT quand la libellule est
        # la menace active (besoin d'élan latéral). Sinon plancher bas
        # pour pouvoir pivoter franchement (roue interne proche de 0).
        near_dragonfly = self.dragonfly_hold > 0
        floor = self.drive_floor if near_dragonfly else 0.05
        drive = np.clip(drive, floor, self.drive_ceiling)

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