"""Render recorded state arrays from either experiment; no inference or training needed."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--trial", type=int, default=0)
    p.add_argument("--stride", type=int, default=2)
    args = p.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    import imageio.v2 as imageio
    import mujoco
    from instanttrack.robot import get_robot
    robot = get_robot("g1")
    data = np.load(args.input)
    state = data["state"]
    if not 0 <= args.trial < state.shape[1] or args.stride < 1:
        raise ValueError("Invalid trial or stride")
    model = mujoco.MjModel.from_xml_path(robot.mjcf_path)
    mjdata = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 135., -15., 3.
    qadr = [model.jnt_qposadr[model.joint(n).id] for n in robot.joint_names]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        with imageio.get_writer(output, fps=50 / args.stride, codec="libx264") as writer:
            for s in state[::args.stride, args.trial]:
                mjdata.qpos[:7] = s[:7]
                mjdata.qpos[qadr] = s[13:42]
                mujoco.mj_forward(model, mjdata)
                camera.lookat[:] = [s[0], s[1], .8]
                renderer.update_scene(mjdata, camera=camera)
                if "targets" in data:
                    scene = renderer.scene
                    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                       np.array([.035, .035, .035]), data["targets"][args.trial],
                                       np.eye(3).flatten(), np.array([1., .15, .15, 1.]))
                    scene.ngeom += 1
                writer.append_data(renderer.render())
    print(output)


if __name__ == "__main__":
    main()
