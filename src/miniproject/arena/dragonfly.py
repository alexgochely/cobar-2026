import uuid
from functools import cached_property, cache
import mujoco
from dm_control.utils import transformations


class DragonFlyMixin:
    @cached_property
    def _dragonflies(self):
        return []
    
    @cached_property
    def _dragonfly_geoms(self):
        return []

    def add_dragonfly(self, scale=0.5):
        name = uuid.uuid4().hex
        parent = self.mjcf_root.worldbody
        dragonfly = parent.add("body", name=name, pos=(0, 0, 0), mocap=True)
        thorax_geom = dragonfly.add(
            "geom",
            name=f"{name}_thorax",
            type="capsule",
            size=(2 * scale,),
            fromto=(-1.25 * scale, 0, 0, 1.25 * scale, 0, 0),
            rgba=(0.15, 0.55, 0.2, 1),
        )
        head_geom = dragonfly.add(
            "geom",
            name=f"{name}_head",
            type="sphere",
            size=(1.6 * scale,),
            pos=(4 * scale, 0, 0.5 * scale),
            rgba=(0.15, 0.55, 0.2, 1),
        )
        dragonfly.add(
            "geom",
            name=f"{name}_eye_l",
            type="sphere",
            size=(1 * scale,),
            pos=(4.6 * scale, 1.2 * scale, 0.9 * scale),
            rgba=(0.6, 0.1, 0.1, 1),
        )
        dragonfly.add(
            "geom",
            name=f"{name}_eye_r",
            type="sphere",
            size=(1 * scale,),
            pos=(4.6 * scale, -1.2 * scale, 0.9 * scale),
            rgba=(0.6, 0.1, 0.1, 1),
        )
        abdomen_geom = dragonfly.add(
            "geom",
            name=f"{name}_abdomen",
            type="capsule",
            size=(1 * scale,),
            fromto=(-3 * scale, 0, 0, -18 * scale, 0, 0),
            rgba=(0.1, 0.4, 0.7, 1),
        )
        wing_l = dragonfly.add("body", name=f"{name}_wing_l", pos=(0, 2 * scale, 0))
        wing_l_geom = wing_l.add(
            "geom",
            name=f"{name}_wing_l",
            type="box",
            size=(2.4 * scale, 6 * scale, 0.08 * scale),
            pos=(0, 6 * scale, 0),
            rgba=(0.7, 0.9, 1.0, 0.35),
        )
        wing_r = dragonfly.add("body", name=f"{name}_wing_r", pos=(0, -2 * scale, 0))
        wing_r_geom = wing_r.add(
            "geom",
            name=f"{name}_wing_r",
            type="box",
            size=(2.4 * scale, 6 * scale, 0.08 * scale),
            pos=(0, -6 * scale, 0),
            rgba=(0.7, 0.9, 1.0, 0.35),
        )
        self.ground_contact_geoms.extend([thorax_geom, head_geom, abdomen_geom])
        self._dragonflies.append(dragonfly)
        self._dragonfly_head = head_geom
        self._dragonfly_geoms.append(
            {
                "thorax": thorax_geom,
                "head": head_geom,
                "abdomen": abdomen_geom,
                "wing_l": wing_l_geom,
                "wing_r": wing_r_geom
            }
        )
        return dragonfly

    @cache
    def _get_dragonfly_mocap_id(self, sim, dragonfly_idx=0):
        dragonfly = self._dragonflies[dragonfly_idx]
        return sim.mj_model.body_mocapid[
            mujoco.mj_name2id(
                sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, dragonfly.full_identifier
            )
        ]

    def set_dragonfly_pose(self, sim, pos, euler_ZYX=None, dragonfly_idx=0):
        mocap_id = self._get_dragonfly_mocap_id(sim, dragonfly_idx)
        sim.mj_data.mocap_pos[mocap_id] = pos
        if euler_ZYX is not None:
            quat = transformations.euler_to_quat(euler_ZYX, "ZYX")
            sim.mj_data.mocap_quat[mocap_id] = quat

    def set_dragonfly_rgba(self, sim, rgba, dragonfly_idx=0, segment="thorax"):
        geom = sim.mj_model.geom(self._dragonfly_geoms[dragonfly_idx][segment].name)
        geom.rgba[:] = rgba
