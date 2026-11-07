"""Fixed-camera rink videos from recorded simulation. No policy inference or training."""
from __future__ import annotations

import argparse
import json
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
    from PIL import Image, ImageDraw, ImageFont
    from .physics import ASSET, build_asset
    if not ASSET.exists():
        build_asset()
    data = np.load(args.input)
    state = data["state"]
    if not 0 <= args.trial < state.shape[1] or args.stride < 1:
        raise ValueError("Invalid trial or frame stride")
    trajectory = state[:, args.trial]
    if not np.isfinite(trajectory).all():
        raise ValueError("Recording contains invalid states; do not render them as a successful demo")
    suite, mode = str(data["suite"]), str(data["mode"])
    spec = mujoco.MjSpec.from_file(str(ASSET))
    spec.add_texture(name="winter_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[.70, .84, .95], rgb2=[.96, .98, 1.], width=256, height=1536)
    spec.worldbody.add_light(name="rink_light", pos=[3, -2, 8], dir=[0, 0, -1],
                            diffuse=[.8, .8, .8], ambient=[.25, .25, .25], castshadow=1)
    spec.worldbody.add_geom(name="rink", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[30, 30, .1], rgba=[.65, .82, .9, 1])
    model = spec.compile()
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 720
    model.vis.headlight.active = 1
    model.vis.headlight.ambient[:] = [.3, .3, .3]
    model.vis.headlight.diffuse[:] = [.7, .7, .7]
    mjdata = mujoco.MjData(model)
    from instanttrack.robot import get_robot
    robot = get_robot("g1")
    qadr = [model.jnt_qposadr[model.joint(n).id] for n in robot.joint_names]
    foot_ids = [model.body(f"{side}_ankle_roll_link").id for side in ("left", "right")]
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation = 125., -25.
    if suite == "course":
        camera.lookat[:] = [3.7, 0., .5]
        camera.distance = 8.5
    else:
        lo, hi = trajectory[:, :2].min(0), trajectory[:, :2].max(0)
        camera.lookat[:] = [*(.5 * (lo + hi)), .65]
        camera.distance = max(4., min(25., 2. + 1.2 * float((hi - lo).max())))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
    except OSError:
        font = ImageFont.load_default()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    trails = [[], []]
    with mujoco.Renderer(model, height=720, width=1280) as renderer:
        def geom(kind, size, pos, rgba):
            scene = renderer.scene
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], kind, np.asarray(size), np.asarray(pos),
                               np.eye(3).flatten(), np.asarray(rgba))
            scene.ngeom += 1
        with imageio.get_writer(out, fps=50 / args.stride, codec="libx264") as writer:
            for frame in range(0, len(trajectory), args.stride):
                s = trajectory[frame]
                mjdata.qpos[:7] = s[:7]
                mjdata.qpos[qadr] = s[13:42]
                mujoco.mj_forward(model, mjdata)
                renderer.update_scene(mjdata, camera=camera)
                for j, body in enumerate(foot_ids):
                    point = mjdata.xpos[body].copy()
                    if frame % 10 == 0 and point[2] < .12:
                        trails[j].append([point[0], point[1], .004])
                    for marker in trails[j][-160:]:
                        geom(mujoco.mjtGeom.mjGEOM_SPHERE, [.008] * 3, marker, [.12, .35, .55, .65])
                passed = int(data["passed"][frame, args.trial])
                if suite == "course":
                    for index, (x, y) in enumerate(data["gates"][args.trial]):
                        color = [.1, .7, .3, 1.] if index < passed else [1., .35, .1, 1.]
                        for side in (-1, 1):
                            geom(mujoco.mjtGeom.mjGEOM_CYLINDER, [.025, .35, 0.], [x, y + .6 * side, .35], color)
                    x, y = data["gates"][args.trial, -1]
                    for angle in np.linspace(0, 2 * np.pi, 48, endpoint=False):
                        geom(mujoco.mjtGeom.mjGEOM_SPHERE, [.018] * 3,
                             [x + .75 * np.cos(angle), y + .75 * np.sin(angle), .005], [.1, .6, .2, 1.])
                image = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 1280, 78), fill=(15, 24, 35))
                speed = float(np.linalg.norm(s[7:9]))
                draw.text((18, 10), f"Blade-contact simulation | {mode} | trial {args.trial} | {frame * .02:.1f}s", font=font, fill="white")
                label = (f"Standing balance | {str(data['trial_name'][args.trial])}" if suite == "balance" else
                         f"Pretrained motion | {str(data['trial_name'][args.trial])}" if suite == "motion" else
                         "Straight skating" if suite == "straight" else f"Gates {passed}/3" if suite == "course" else
                         ["Forward", "Left turn", "Right turn", "Accelerate then brake"][args.trial])
                draw.text((18, 42), f"{label} | speed {speed:.2f} m/s | fixed camera", font=font, fill="white")
                if "terminated" in data and data["terminated"][frame, args.trial]:
                    draw.rectangle((0, 78, 1280, 110), fill=(125, 20, 20))
                    draw.text((18, 82), "TERMINATED: fall or numerical failure", font=font, fill="white")
                writer.append_data(np.asarray(image))
    print(out)


if __name__ == "__main__":
    main()
