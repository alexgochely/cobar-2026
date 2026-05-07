import numpy as np
from miniproject.simulation import MiniprojectSimulation


class Controller:
    """Level 0-2 : navigation olfactive + évitement visuel d'obstacles.
    
    Architecture hiérarchique :
    - Haut niveau (cerveau) : olfaction → drive attractif vers banane
                              vision   → drive répulsif des obstacles
    - Bas niveau (VNC) : TurningController → angles articulaires + adhésion
    
    Si un obstacle est vu → l'évitement remplace l'olfaction (subsumption).
    Si la mouche est sur la banane → elle s'arrête.
    """

    def __init__(self, sim: MiniprojectSimulation):
        from flygym.examples.locomotion import TurningController
        self.turning_controller = TurningController(sim.timestep)

        # === Gains olfactifs ===
        self.attractive_gain = -1000.0
        self.aversive_gain = 80.0

        # === Paramètres de vision ===
        self.obj_threshold = 0.30        # pixel plus sombre = obstacle
        self.obstacle_area_thr = 0.20    # aire min pour déclencher l'évitement
        self.visual_gain = 10.0           # force de la répulsion

        # === Paramètres du vent ===
        self.wind_gain = 1.0           # placeholder, à tuner sur Level 3
        self.wind_threshold = 0.5      # > 0.28 (mesuré sans vent) pour éviter faux positifs

        # === Timing des décisions === 
        self.decision_interval = 0.05    # secondes
        self._last_decision_time = -np.inf
        self._last_control_signal = np.array([1.0, 1.0])

        # === Arrêt sur la nourriture ===
        self.stop_distance = 2.0
        self.target_xy = np.array(sim.world.banana_xy)

        # === Pré-calcul des centres de masse des ommatidies ===
        self.fly = sim.fly
        n_ommatidia = self.fly.retina.num_ommatidia_per_eye
        self.coms = np.empty((n_ommatidia, 2))
        for i in range(n_ommatidia):
            mask = self.fly.retina.ommatidia_id_map == i + 1
            self.coms[i] = np.argwhere(mask).mean(axis=0)

        # Masque : moitié haute de la rétine (exclut le sol)
        self.upper_mask = self.coms[:, 0] < (self.fly.retina.nrows / 2)
        self.n_upper = self.upper_mask.sum()

    def _compute_olfactory_drive(self, odor_intensities):
        """Drive basé sur l'odeur (attire vers banane)."""
        attractive = np.average(
            odor_intensities[:, 0].reshape(2, 2), axis=0, weights=[9, 1]
        )
        attractive_bias = 0.0
        if attractive.mean() > 0:
            attractive_bias = (
                self.attractive_gain
                * (attractive[0] - attractive[1])
                / attractive.mean()
            )

        aversive_bias = 0.0
        if odor_intensities.shape[1] > 1:
            aversive = np.average(
                odor_intensities[:, 1].reshape(2, 2), axis=0, weights=[10, 0]
            )
            if aversive.mean() > 0:
                aversive_bias = (
                    self.aversive_gain
                    * (aversive[0] - aversive[1])
                    / aversive.mean()
                )

        effective_bias = attractive_bias + aversive_bias
        effective_bias_norm = np.tanh(effective_bias ** 2) * np.sign(effective_bias)

        drive = np.ones(2)
        side_to_modulate = int(effective_bias_norm > 0)
        drive[side_to_modulate] -= np.abs(effective_bias_norm) * 0.75
        return drive

    def _detect_obstacles(self, ommatidia_readouts):
        """Retourne (area_L, area_R, avoidance_active)."""
        left_eye = ommatidia_readouts[0]
        right_eye = ommatidia_readouts[1]

        # Ne regarde que la moitié haute (exclut le sol qui apparaît sombre en bas)
        is_obj_L = (left_eye.max(axis=1) < self.obj_threshold) & self.upper_mask
        is_obj_R = (right_eye.max(axis=1) < self.obj_threshold) & self.upper_mask

        area_L = is_obj_L.sum() / self.n_upper
        area_R = is_obj_R.sum() / self.n_upper

        avoidance_active = (area_L > self.obstacle_area_thr) or \
                           (area_R > self.obstacle_area_thr)
        return area_L, area_R, avoidance_active
    
    def _compute_wind_drive(self, antenna_data):
        """Drive basé sur le vent (oriente la mouche face au vent = upwind).
        Retourne (drive (direction (2,)), wind_detected (bool de detection du vent)).
        """
        # Garde pour chaque coté les composantes de deviation (sans w)
        deviation_l = antenna_data['l']['qpos'][1:]
        deviation_r = antenna_data['r']['qpos'][1:]

        # Addition des déviations de chaque antennes et controle si vent obtenu est significatif
        wind_magnitude = (np.linalg.norm(deviation_l) + np.linalg.norm(deviation_r))
        wind_detected = wind_magnitude > self.wind_threshold

        if not wind_detected:
            return np.ones(2), False

        # Calcul du biais du vent et normalisation
        wind_bias = self.wind_gain * (deviation_l[2] + deviation_r[2])
        effective_bias_norm = np.tanh(wind_bias ** 2) * np.sign(wind_bias)
        assert np.sign(effective_bias_norm) == np.sign(wind_bias)

        drive = np.ones(2)
        side_to_modulate = int(effective_bias_norm > 0)
        modulation_amount = np.abs(effective_bias_norm) * 0.9
        drive[side_to_modulate] -= modulation_amount

        return drive, wind_detected
    

    def _compute_control_signal(self, odor_intensities, ommatidia_readouts, antenna_data):
        """Fusionne olfaction + vent + vision. L'évitement a priorité sur le vent qui a priorité sur l'attraction."""
        
        # Drive olfactif (comportement par défaut : va vers la banane)
        drive = self._compute_olfactory_drive(odor_intensities)

        # Drive du vent (écrase l'olfaction si detecté)
        wind_drive, wind_detected = self._compute_wind_drive(antenna_data)
        if wind_detected:
            drive = wind_drive

        # Détection d'obstacles
        area_L, area_R, avoidance_active = self._detect_obstacles(ommatidia_readouts)

        if avoidance_active:
            # Évitement : écrase complètement le drive olfactif (subsumption)
            diff = area_L - area_R
            repulsion = self.visual_gain * abs(diff)

            if diff > 0:  # obstacle à gauche → tourne à droite
                drive[0] = max(1.0 - repulsion, 0.1)
                drive[1] = 1.2
            elif diff < 0:  # obstacle à droite → tourne à gauche
                drive[1] = max(1.0 - repulsion, 0.1)
                drive[0] = 1.2
            else:
                # Obstacle pile devant → tourne à droite arbitrairement
                total = area_L + area_R
                drive[0] = max(1.0 - self.visual_gain * total * 0.5, 0.1)
                drive[1] = 1.2

        drive = np.clip(drive * 1.15, 0.05, 1.4)
        return drive

    def step(self, sim: MiniprojectSimulation):
        # Arrêt sur la banane
        fly_pos = sim.mj_data.body(f"{sim.fly.name}/").xpos[:2]
        dist = np.linalg.norm(fly_pos - self.target_xy)
        if dist < self.stop_distance:
            joint_angles, adhesion = self.turning_controller.step(np.array([0.0, 0.0]))
            return joint_angles, adhesion

        # Mise à jour du signal descendant toutes les decision_interval secondes
        current_time = sim.mj_data.time
        if current_time - self._last_decision_time >= self.decision_interval:
            odor = sim.get_olfaction(sim.fly.name)
            ommatidia = sim.get_ommatidia_readouts(sim.fly.name)
            antenna = sim.get_antenna_data(sim.fly.name)
            self._last_control_signal = self._compute_control_signal(odor, ommatidia, antenna)
            self._last_decision_time = current_time

        # VNC : CPG + adhésion
        joint_angles, adhesion = self.turning_controller.step(self._last_control_signal)
        return joint_angles, adhesion