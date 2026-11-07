# Skate gait adaptation pilot

This pilot fine-tunes the original full G1 motion-tracking controller on the derived skate asset. It selects up to three named `walk` clips whose root paths are the straightest, raises their references to the blade height, and runs the same references under directional blade friction.

The learned world model starts from the original checkpoint's shared trunk. A new contact-mode output head is initialized for skate contacts and trained during a 1,000-iteration world-model-only phase while the pretrained policy collects skate transitions. Then policy and critic fine-tuning starts at a low learning rate. The reward retains motion tracking and adds a supported blade-axis glide term that suppresses lateral blade motion. Checkpoints are written every two minutes during a 30-minute pilot.

The run automatically evaluates the fine-tuned checkpoint on the same selected clips and records tracking error, forward speed, blade contact, longitudinal blade speed, lateral motion, falls, and videos. This is a development pilot on its training motions, not held-out validation. Because the motion prior is walking, the result may still look walk-like; inspect it before calling it a skating gait.

On the 4090:

```bash
bash scripts/skate_gait_adapt.sh
```

Output goes to `artifacts/skate_gait_adapt_v1/`; the training checkpoint is under `third_party/instantTrack/logs/skate_gait_adapt_v1/`. Existing directories are preserved. The script uses the pinned upstream trainer in memory and does not edit the vendor checkout.
