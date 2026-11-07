"""Test blade friction at two headings in CPU MuJoCo and optional MuJoCo Warp.

The probe has translation joints and fixed orientation to measure slip, not rolling.
No force is applied after its initial velocity is set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from .physics import LONG_FRICTION, SIDE_FRICTION


def probe_xml(yaw):
    return f'''<mujoco><option timestep=".005" integrator="implicitfast" cone="pyramidal" iterations="100"/>
      <worldbody><geom name="ice" type="plane" size="5 5 .1"/>
      <body pos="0 0 .0145" euler="0 0 {yaw}">
      <joint type="slide" axis="1 0 0"/><joint type="slide" axis="0 1 0"/>
      <joint type="slide" axis="0 0 1"/>
      <geom name="blade" type="capsule" fromto="-.1 0 0 .1 0 0" size=".015" mass="1" contype="0" conaffinity="0"/>
      </body></worldbody><contact><pair geom1="blade" geom2="ice" condim="3"
      friction="{LONG_FRICTION} {SIDE_FRICTION} .001 .0001 .0001" solref=".01 1"/></contact></mujoco>'''


def trial(yaw, axis, backend="mujoco"):
    m = mujoco.MjModel.from_xml_string(probe_xml(yaw))
    d = mujoco.MjData(m)
    for _ in range(40):
        mujoco.mj_step(m, d)
    d.qvel[:] = 0
    d.qvel[axis] = 1.
    start = d.qpos.copy()
    mujoco.mj_forward(m, d)
    direction = d.geom_xmat[m.geom("blade").id].reshape(3, 3)[:, 2]
    tangent = d.contact[0].frame.reshape(3, 3)[1]
    alignment = abs(float(tangent @ direction))
    if backend == "mujoco":
        for _ in range(100):
            mujoco.mj_step(m, d)
        end = d.qpos
    else:
        import warp as wp
        import mujoco_warp as mjw
        with wp.ScopedDevice(backend):
            wm = mjw.put_model(m)
            wd = mjw.put_data(m, d, nworld=1)
            for _ in range(100):
                mjw.step(wm, wd)
            end = wd.qpos.numpy()[0]
    return dict(displacement_m=float(end[axis] - start[axis]), tangent_alignment=alignment)


def check(backend):
    rows = []
    for yaw in (0, 90):
        along, across = trial(yaw, 0, backend), trial(yaw, 1, backend)
        ratio = abs(along["displacement_m"]) / max(abs(across["displacement_m"]), 1e-8)
        if along["tangent_alignment"] < .99 or ratio < 5 or along["displacement_m"] < .2:
            raise AssertionError(f"Directional contact failed: {yaw=}, {along=}, {across=}, {ratio=}")
        rows.append(dict(yaw_deg=yaw, along=along, across=across, displacement_ratio=ratio))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warp", choices=["cpu", "cuda:0"])
    p.add_argument("--output")
    args = p.parse_args()
    results = {"mujoco_cpu": check("mujoco")}
    if args.warp:
        results["mujoco_warp_" + args.warp] = check(args.warp)
    text = json.dumps(results, indent=2)
    print(text)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")


if __name__ == "__main__":
    main()
