from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco

from experiments.wm_control.common import ROOT

ASSET = ROOT / "artifacts/skating_assets/g1_skates.xml"
LONG_FRICTION = .025
SIDE_FRICTION = 1.2
BLADE_Z = -.055
BLADE_RADIUS = .006
ANKLE_HEIGHT = -BLADE_Z + BLADE_RADIUS


def build_asset():
    """New derived MJCF; the original G1 asset and installed environment remain unchanged."""
    from instanttrack.robot import get_robot
    original = Path(get_robot("g1").mjcf_path)
    if not original.is_file():
        raise FileNotFoundError(f"Existing G1 asset missing: {original}. Use your original checkout/environment.")
    tree = ET.parse(original)
    root = tree.getroot()
    for mesh in root.findall("asset/mesh"):
        mesh.set("file", str((original.parent / mesh.attrib["file"]).resolve()))
    for geom in root.findall(".//geom"):
        # Preserve upstream's no-self-collision convention. Explicit blade pairs handle the feet.
        if geom.get("contype", "1") != "0" or geom.get("conaffinity", "1") != "0":
            geom.set("contype", "0")
            geom.set("conaffinity", "1")
    for side in ("left", "right"):
        body = root.find(f".//body[@name='{side}_ankle_roll_link']")
        if body is None:
            raise ValueError("Missing G1 ankle body")
        for geom in body.findall("geom"):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
        ET.SubElement(body, "geom", dict(name=f"{side}_skate_blade", type="capsule",
                      fromto=f"-.06 0 {BLADE_Z} .14 0 {BLADE_Z}", size=str(BLADE_RADIUS),
                      mass="0", contype="0", conaffinity="0", rgba=".7 .85 .95 1", group="0"))
    ASSET.parent.mkdir(parents=True, exist_ok=True)
    tree.write(ASSET, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(ASSET))
    if model.nq != 36 or model.nv != 35:
        raise ValueError("Skates changed the robot state layout")
    return ASSET


def skate_robot():
    from instanttrack.robot import get_robot
    original = get_robot("g1")
    if not ASSET.exists():
        build_asset()
    def cfg():
        return replace(original.robot_cfg(), spec_fn=lambda: mujoco.MjSpec.from_file(str(ASSET)))
    # Blade pairs have fixed coefficients. Preserve the original context range/normalization;
    # geom friction randomization affects ordinary body-floor contacts, not the blade pairs.
    dr = replace(original.dr, reset_lin_vel=(.05, .05, .02),
                 reset_ang_vel=(.05, .05, .10), reset_joint_pos=.03,
                 reset_rot=(.03, .03, .05), reset_pos=(.02, .02, .005))
    return replace(original, mjcf_path=str(ASSET), robot_cfg=cfg, dr=dr)


def configure_scene(spec):
    planes = [g for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_PLANE]
    if len(planes) != 1:
        raise ValueError("Expected exactly one rink plane")
    floor = planes[0]
    floor.rgba = [.72, .86, .92, 1.]
    for side in ("left", "right"):
        spec.add_pair(name=f"{side}_blade_ice", geomname1=f"robot/{side}_skate_blade", geomname2=floor.name,
                      condim=3, friction=[LONG_FRICTION, SIDE_FRICTION, .001, .0001, .0001],
                      solref=[.01, 1.])


def install_scene():
    """Patch the scene factory in this process only, before constructing the simulator."""
    import instanttrack.env as env_module
    old = env_module.make_scene_cfg
    if getattr(old, "_skating", False):
        return
    def scene(robot, num_envs):
        return replace(old(robot, num_envs), spec_fn=configure_scene)
    scene._skating = True
    env_module.make_scene_cfg = scene
