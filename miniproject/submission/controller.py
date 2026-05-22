import numpy as np
from miniproject.simulation import MiniprojectSimulation


class Controller:
    """
    Controller du mini-projet COBAR.

    Architecture reactive de subsumption :
        escape (libellule) > danger (herbe) > wind > olfaction

    DETECTION DE LA LIBELLULE
    -------------------------
    La libellule est composee de geometries colorees (cf src/miniproject/
    arena/dragonfly.py) :
      - thorax + tete : vert fonce (0.15, 0.55, 0.2)
      - yeux         : rouge       (0.6,  0.1, 0.1)
      - abdomen      : BLEU FONCE  (0.1,  0.4, 0.7)  <-- signature visuelle
      - ailes        : bleu clair semi-transparent (0.7, 0.9, 1.0, 0.35)

    Le bleu fonce de l'abdomen est la signature la plus distinctive : le
    ciel est bleu clair (B >> R), et il n'y a aucun autre objet bleu fonce
    dans la scene (sol et herbe sont verts, banane est jaune). On detecte
    donc des pixels ou B est dominant ET R/G/B forment l'equivalent du
    bleu sature de l'abdomen.

    L'esquive utilise la latéralisation visuelle (gauche/droite) plutot
    que la position 3D exacte : la mouche fuit du cote oppose a celui ou
    elle voit le plus de "bleu libellule".
    """

    def __init__(self, sim: MiniprojectSimulation):
        from flygym.examples.locomotion import TurningController
        self.turning_controller = TurningController(sim.timestep)

        # ── Olfaction ─────────────────────────────────────────
        self.attractive_gain = -1000.0
        self.aversive_gain = 10.0

        # ── Vision herbe (ROI) ────────────────────────────────
        self.roi_top = 0.0
        self.roi_bottom = 0.4

        self.danger_thr = 0.035
        self.danger_zone_width = 0.5
        self.danger_speed = 1
        self.danger_turn = 0.05

        self.danger_hold_sec = 0.3
        self.danger_hold_max = int(self.danger_hold_sec / 0.02)
        self.danger_hold = 0

        # ── Lissage commande ─────────────────────────────────
        self.drive_smoothing = 0.0
        self.previous_drive = np.array([1.0, 1.0])

        # ── Vent ─────────────────────────────────────────────
        self.wind_gain = 1.0
        self.wind_threshold = 0.5

        # ── Detection libellule (visuelle, par couleur) ───────
        # ROI plus large que pour l'herbe : la libellule peut etre n'importe
        # ou (en l'air -> haut de l'image ; en plongeon -> milieu).
        self.dfly_roi_top = 0.0
        self.dfly_roi_bottom = 0.55      # presque toute la moitie haute
        # Seuils RGB pour le bleu fonce de l'abdomen (rgba ~ 0.1, 0.4, 0.7).
        # En image normalisee 0-255 : R~25, G~100, B~178.
        # On accepte une marge large pour la libellule au loin (mix de couleurs).
        self.dfly_b_min = 90             # B doit etre eleve
        self.dfly_b_over_r = 1.6         # B > 1.6*R  (exclut blanc/cyan/ciel pur)
        self.dfly_b_over_g = 1.3         # B > 1.3*G
        # Seuil de declenchement : fraction de pixels "libellule" dans le ROI.
        self.dfly_thr = 0.002            # 0.2% du ROI = quelques pixels suffisent
        # Hold : une fois detectee, on reste en mode esquive ~1s meme si
        # elle disparait temporairement du champ visuel.
        self.dfly_hold_sec = 1.0
        self.dfly_hold_max = int(self.dfly_hold_sec / 0.02)
        self.dfly_hold = 0
        # Esquive : memorise le cote oppose a la libellule au moment de la
        # detection. Sans connaitre la position 3D, on se contente de fuir
        # lateralement du cote ou la libellule n'est PAS.
        self._escape_dir = 0   # +1 = fuir a gauche, -1 = a droite, 0 = pas decide
        self.escape_speed = 1.2     # vitesse de fuite (dash)
        self.escape_turn  = 0.0     # roue cote esquive : forte rotation

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
        self._last_dfly_L = 0.0
        self._last_dfly_R = 0.0
        self._last_mode = "olfaction"

        self.fly = sim.fly

    # ════════════════════════════════════════════════════════════
    # VISION : detection libellule (signature bleu fonce de l'abdomen)
    # ════════════════════════════════════════════════════════════
    def _detect_dragonfly(self, raw_vision):
        """Retourne (frac_L, frac_R) : fraction de pixels 'bleu libellule'
        dans le ROI haut de chaque oeil."""
        fracs = []
        for img in raw_vision:
            H, _, _ = img.shape
            roi = img[int(H * self.dfly_roi_top):int(H * self.dfly_roi_bottom),
                      :, :]
            r = roi[..., 0].astype(np.float32)
            g = roi[..., 1].astype(np.float32)
            b = roi[..., 2].astype(np.float32)
            blue_pixel = (
                (b > self.dfly_b_min)
                & (b > self.dfly_b_over_r * r)
                & (b > self.dfly_b_over_g * g)
            )
            fracs.append(float(blue_pixel.mean()))
        return fracs[0], fracs[1]

    # ════════════════════════════════════════════════════════════
    # VISION : detection herbe (couleur verte par moyenne)
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
            if i == 0:  # oeil gauche -> bord droit
                dangers.append(grass[:, -zone_w:].mean())
            else:       # oeil droit -> bord gauche
                dangers.append(grass[:, :zone_w].mean())
        return dangers[0], dangers[1]

    def _visual_analysis(self, raw_vision, pref_left):
        # Herbe (la libellule est geree separement dans _compute_control_signal)
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
                turn_left = diff < 0
            if turn_left:
                drive = np.array([self.danger_turn, self.danger_speed])
            else:
                drive = np.array([self.danger_speed, self.danger_turn])
            return "danger", drive

        return "clear", None

    # ════════════════════════════════════════════════════════════
    # LIBELLULE : esquive purement visuelle
    # ════════════════════════════════════════════════════════════
    def _escape_drive(self, raw_vision):
        """
        Esquive purement visuelle. On a deja decide _escape_dir au moment de
        la detection initiale (cote oppose au cote ou on voyait la libellule).
        On fonce vite + virage serre vers ce cote.

        Si en cours d'esquive on voit de l'herbe dans le cote choisi, on
        diminue legerement la rotation pour ne pas se planter dans un brin.
        """
        # Drive de base : tourner du cote _escape_dir (+1 = gauche, -1 = droite)
        # tout en gardant une vitesse elevee (foncer).
        if self._escape_dir > 0:
            # Fuir a gauche -> roue gauche lente, roue droite rapide
            drive = np.array([self.escape_turn, self.escape_speed])
        else:
            # Fuir a droite -> roue gauche rapide, roue droite lente
            drive = np.array([self.escape_speed, self.escape_turn])

        # Petit ajustement : si pendant la fuite on voit beaucoup d'herbe du
        # cote ou on fuit, on attenue la rotation (sinon on fonce dans l'herbe)
        dL, dR = self._detect_grass(raw_vision)
        if self._escape_dir > 0 and dL > 2 * self.danger_thr:
            # on fuit a gauche mais y a de l'herbe a gauche -> moins serre
            drive[0] = 0.4
        elif self._escape_dir < 0 and dR > 2 * self.danger_thr:
            drive[1] = 0.4

        return drive

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

        pref_left = drive[0] < drive[1]

        wind_drive, wind_detected = self._wind_drive(antenna)
        if wind_detected:
            drive = 0.5 * drive + 0.5 * wind_drive
            self._last_mode = "wind"

        # ── Libellule : detection visuelle prioritaire ──
        dfly_L, dfly_R = self._detect_dragonfly(raw_vision)
        self._last_dfly_L, self._last_dfly_R = dfly_L, dfly_R
        dfly_max = max(dfly_L, dfly_R)

        if dfly_max > self.dfly_thr:
            # Detection courante : on (re)initialise le hold + le cote d'esquive
            if self.dfly_hold == 0:
                # debut d'une attaque : choisir le cote oppose
                # libellule a gauche (dfly_L > dfly_R) -> fuir a droite -> dir=-1
                # libellule a droite -> fuir a gauche -> dir=+1
                self._escape_dir = -1 if dfly_L > dfly_R else +1
            self.dfly_hold = self.dfly_hold_max

        if self.dfly_hold > 0:
            self.dfly_hold -= 1
            drive = self._escape_drive(raw_vision)
            self._last_mode = "escape"
        else:
            # Pas d'attaque -> comportement normal (herbe puis olfaction)
            self._escape_dir = 0
            action, payload = self._visual_analysis(raw_vision, pref_left)
            if action == "danger":
                drive = payload
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
        body = sim.mj_data.body(f"{sim.fly.name}/")
        fly_xy = np.array(body.xpos[:2], dtype=float)
        if np.linalg.norm(fly_xy - self.target_xy) < self.stop_distance:
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
