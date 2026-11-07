"""Fine-tune the pretrained G1 motion tracker for directional blade contacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from experiments.wm_control.common import UPSTREAM, check_upstream, file_hash, load_checkpoint
from experiments.wm_control.train_reach import patch_once
from .physics import ANKLE_HEIGHT, install_scene, skate_robot
from .train import training_source


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Full original motion-tracking checkpoint")
    parser.add_argument("--clip_regex", default="walk")
    parser.add_argument("--max_clips", type=int, default=3)
    parser.add_argument("--glide_weight", type=float, default=1.0)
    parser.add_argument("--wm_warmup", type=int, default=1000)
    parser.add_argument("--device", choices=["cpu", "cuda:0"], default="cuda:0")
    parser.add_argument("--diagnose", action="store_true")
    args, training_args = parser.parse_known_args()
    if args.max_clips < 1 or args.wm_warmup < 1 or args.glide_weight < 0:
        parser.error("Invalid clip count, warmup, or glide weight")
    if "--log_dir" not in training_args:
        parser.error("Pass a fresh --log_dir")
    if "--resume" in training_args:
        parser.error("Start a fresh adaptation run; do not resume another task")
    for flag in ("--wm_only_iters", "--wm_contact", "--wm_foot_weight", "--contact_reward"):
        if flag in training_args:
            parser.error(f"{flag} is set by this skating protocol")
    args.source = str(Path(args.source).resolve())
    log_index = training_args.index("--log_dir") + 1
    log_dir = Path(training_args[log_index]).resolve()
    training_args[log_index] = str(log_dir)
    if log_dir.exists():
        raise FileExistsError(f"Preserving existing run: {log_dir}")
    check_upstream()

    import instanttrack.robot as robot_module
    import instanttrack.mdp as mdp_module
    import instanttrack.networks as networks
    from instanttrack.motion import MotionLib
    from .gait_adaptation import SkateMotionMDP, selected_motion_library, load_compatible_world_model

    robot = skate_robot()
    # The original reset perturbations and pushes are retained for robust data collection.
    install_scene()
    robot_module.get_robot = lambda name: robot if name == "g1" else (_ for _ in ()).throw(ValueError("G1 only"))
    mdp_module.MDP = SkateMotionMDP
    SkateMotionMDP.default_glide_weight = args.glide_weight
    contact_z = ANKLE_HEIGHT + .025
    # The ankle-roll origin is raised to put the blade on the ice; update both the reward
    # and contact-mode labels to use the skate asset's corresponding foot height.
    mdp_module.CONTACT_Z = contact_z
    mdp_module.MAX_STEP_REWARD = (5. + args.glide_weight) * .02
    selected_meta = []
    def make_motion(path, spec, kin, device):
        nonlocal selected_meta
        motion, selected_meta = selected_motion_library(path, spec, kin, device,
                                                         args.clip_regex, args.max_clips)
        return motion
    MotionLib.from_dir = staticmethod(make_motion)

    source = (UPSTREAM / "src/instanttrack/train.py").read_text()
    source = training_source(source, frozen=False, diagnose=args.diagnose)
    source = patch_once(source, "FOOT_CONTACT_Z = 0.045", f"FOOT_CONTACT_Z = {contact_z:.6f}")
    for flag in ("--cuda_graph", "--cudagraphs", "--wm_ens", "--pi_simba", "--pi_experts", "--ref_residual"):
        if flag in training_args:
            raise ValueError(f"Unsupported option in this skating adaptation: {flag}")
    training_args += ["--wm_only_iters", str(args.wm_warmup), "--wm_contact", ".05",
                      "--wm_foot_weight", ".1"]
    task = dict(name="skate_motion_adaptation", source_checkpoint=args.source,
                source_sha256=file_hash(args.source), selected_clips=selected_meta,
                glide_weight=args.glide_weight, glide_definition="supported blade-axis speed with lateral-slip suppression",
                blade_long_friction=.025, blade_lateral_friction=1.2,
                contact_height_m=contact_z, wm_contact_modes=True, wm_warmup=args.wm_warmup,
                protocol="fine-tune the pretrained motion-tracking policy and dynamics on straight walk references with skate contacts")

    def transfer_init(ns):
        ck = load_checkpoint(args.source, device=ns["dev"])
        for name in ("wm_hidden", "enc_hidden", "dec_hidden", "hist", "goal_frames"):
            if ck["args"].get(name) != getattr(ns["args"], name):
                raise ValueError(f"Source and skate adaptation architecture disagree on {name}")
        if ck["args"].get("ref_residual", 0) or ck["args"].get("pi_experts", 1) != 1 or ck["args"].get("pi_simba", ""):
            raise ValueError("Expected the original SuperMimic+EMA motion-tracking checkpoint")
        ns["pi"].load_state_dict(ck["pi"])
        copied, skipped = load_compatible_world_model(ns["wm"], ck["wm"])
        ns["critic"].load_state_dict(ck["critic"])
        if ns["pi_avg"] is not None:
            ns["pi_avg"].module.load_state_dict(ns["pi"].state_dict())
        task.update(world_model_copied_keys=copied, world_model_skipped_source_keys=skipped)
        (log_dir / "task.json").write_text(json.dumps(task, indent=2) + "\n")
        print(f"Loaded pretrained actor, critic, and compatible world-model trunk; new contact head keys: {skipped}", flush=True)

    source = patch_once(source, 'dev = "cuda:0"', f'dev = {args.device!r}')
    sys.argv = [str(UPSTREAM / "src/instanttrack/train.py"), *training_args]
    g = dict(__name__="__main__", __file__=str(UPSTREAM / "src/instanttrack/train.py"),
             _task_metadata=task, _transfer_init=transfer_init)
    exec(compile(source, g["__file__"], "exec"), g)


if __name__ == "__main__":
    main()
