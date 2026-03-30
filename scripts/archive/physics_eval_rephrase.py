"""
Test robustness of physics eval by rephrasing prompts in different ways.

Downloads the best v11 checkpoint (s630) and generates responses for
original + rephrased prompts, saving all outputs for manual comparison.

Usage:
    python -m scripts.physics_eval_rephrase --cache-dir /opt/dlami/nvme/hf_cache
"""

import argparse
import gc
import json
import os

import torch
from huggingface_hub import snapshot_download

from nanochat.checkpoint_manager import load_checkpoint, _patch_missing_config_keys, _patch_missing_keys
from nanochat.engine import Engine
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import RustBPETokenizer


# ---------------------------------------------------------------------------
# Rephrased prompts: 3 variants per task
# ---------------------------------------------------------------------------

REPHRASED_PROMPTS = {
    "uv_catastrophe_main": {
        "noisy": (
            "A heated cavity in thermal equilibrium emits radiation with spectral "
            "energy density u(nu, T), where nu is frequency and T is temperature.\n\n"
            "Observed facts:\n"
            "1. At low frequencies, the emitted energy density increases with frequency.\n"
            "2. For each temperature, the spectrum reaches a maximum at some frequency.\n"
            "3. At higher frequencies, the emitted energy density falls off rapidly toward zero.\n"
            "4. The total emitted energy is finite.\n"
            "5. As temperature increases, the peak shifts to higher frequency and "
            "the total emitted energy increases.\n\n"
            "Assumptions under consideration:\n"
            "1. The electromagnetic field in the cavity can be decomposed into modes.\n"
            "2. The cavity walls are perfectly rigid and do not vibrate.\n"
            "3. The number of modes in a frequency interval grows roughly like nu^2.\n"
            "4. Radiation travels at the speed of light inside the cavity.\n"
            "5. The cavity is in thermal equilibrium with its surroundings.\n"
            "6. If energy is continuous and ordinary thermal equipartition holds, "
            "each mode should have mean energy kT.\n"
            "7. The radiation inside the cavity obeys Maxwell's equations.\n"
            "8. The walls of the cavity absorb and re-emit radiation at all frequencies.\n"
            "9. The cavity is large compared to the wavelengths of interest.\n"
            "10. Gravity has no significant effect on the radiation inside the cavity.\n\n"
            "Under those assumptions, the predicted spectral energy density grows like:\n"
            "u_classical(nu, T) ~ nu^2 T\n\n"
            "This prediction keeps increasing with frequency and leads to an infinite "
            "total energy, in contradiction with experiment.\n\n"
            "Which assumption is most likely wrong? How can we reconcile the prediction "
            "with experiment?"
        ),
    },

    "photoelectric_effect_main": {
        "noisy": (
            "Experimental observations:\n"
            "1. When light shines on a metal, electrons are sometimes ejected.\n"
            "2. Below a certain frequency, no electrons are emitted regardless of brightness.\n"
            "3. Above that threshold, electrons are emitted.\n"
            "4. Brighter light increases the number of electrons but not their maximum energy.\n"
            "5. Higher frequency increases the maximum kinetic energy of electrons.\n"
            "6. Emission occurs with no measurable time delay.\n\n"
            "Assumptions under consideration:\n"
            "1. The metal surface is flat and uniform.\n"
            "2. Light is a continuous wave.\n"
            "3. Electrons in the metal are bound with a characteristic energy.\n"
            "4. The energy delivered by light increases smoothly with intensity.\n"
            "5. The metal is at room temperature.\n"
            "6. A more intense wave should transfer more energy to an electron.\n"
            "7. The experiment is conducted in vacuum.\n"
            "8. If energy delivery is continuous, dim light should accumulate energy "
            "over time and eventually free an electron.\n"
            "9. The electrons interact with one another inside the metal.\n"
            "10. The metal obeys Ohm's law for bulk conduction.\n\n"
            "These assumptions do not match the observations.\n\n"
            "Which assumption is most likely wrong? What can explain the threshold?"
        ),
    },

    "sr_frozen_light": {
        "noisy": (
            "Suppose light is an electromagnetic wave governed by Maxwell's equations, "
            "and suppose also that all inertial observers measure light in vacuum to "
            "have the same speed c.\n\n"
            "Now consider an observer trying to move alongside a light beam.\n\n"
            "Assumptions under consideration:\n"
            "1. Light is a transverse wave.\n"
            "2. Under ordinary velocity addition, the observer should see the light "
            "wave as stationary or nearly frozen.\n"
            "3. The observer carries no electric charge.\n"
            "4. Maxwell's equations hold in every inertial frame.\n"
            "5. The observer's mass does not change with velocity.\n"
            "6. The medium through which light propagates is at rest.\n"
            "7. The observer's clock runs at the same rate regardless of motion.\n"
            "8. Space is three-dimensional and Euclidean.\n\n"
            "Would a frozen light wave make sense? Which assumption leads to trouble?"
        ),
    },

    "sr_approaching_c": {
        "noisy": (
            "Suppose there exists a universal speed c that no material object can exceed.\n\n"
            "Assumptions under consideration:\n"
            "1. Force equals mass times acceleration (F = ma).\n"
            "2. The object is made of ordinary matter.\n"
            "3. Applying more force for longer keeps increasing speed without limit.\n"
            "4. Energy is conserved.\n"
            "5. The object does not emit radiation as it accelerates.\n"
            "6. The mass of the object is constant regardless of speed.\n"
            "7. Space is uniform in all directions.\n"
            "8. The force is applied by contact, not at a distance.\n"
            "9. Time passes at the same rate for all observers.\n\n"
            "As an object approaches c with constant applied force, what happens? "
            "Where does the additional energy go?"
        ),
    },

    "sr_train_lightning": {
        "noisy": (
            "A train moves to the right at constant speed relative to the ground.\n"
            "Two lightning bolts strike the front and back ends of the train.\n\n"
            "A ground observer standing exactly midway between the two strike points "
            "sees the two flashes arrive at the same time and concludes that the "
            "strikes were simultaneous in the ground frame.\n\n"
            "Now consider an observer standing at the midpoint inside the moving train.\n\n"
            "Assumptions under consideration:\n"
            "1. There is no preferred inertial frame.\n"
            "2. Light has the same speed in all inertial frames.\n"
            "3. The train moves at constant velocity (no acceleration).\n"
            "4. The air inside the train moves with the train.\n"
            "5. The lightning bolts are identical in energy.\n"
            "6. Sound travels at a fixed speed relative to the air.\n"
            "7. The train track is perfectly straight and level.\n"
            "8. The ground observer's clocks are synchronized.\n"
            "9. The train is rigid and does not contract or expand.\n"
            "10. Both observers have identical sensory apparatus.\n\n"
            "Using these assumptions, determine what the train observer concludes. "
            "Which flash does the train observer see first? What does this entail?"
        ),
    },

    "sr_michelson_morley": {
        "noisy": (
            "Physicists considered the possibility that light travels through a real "
            "medium filling space.\n\n"
            "Michelson and Morley tested this by comparing light travel times in "
            "perpendicular directions and rotating the apparatus. No shift was observed.\n\n"
            "Assumptions under consideration:\n"
            "1. There exists a preferred rest frame for the medium.\n"
            "2. Earth's motion through the medium creates an effective wind.\n"
            "3. The apparatus arms are of equal length.\n"
            "4. The measured speed of light depends on direction relative to the medium.\n"
            "5. The light source is monochromatic.\n"
            "6. The mirrors are perfectly reflective.\n"
            "7. Temperature does not affect the apparatus during the experiment.\n"
            "8. Maxwell's equations predict light at a fixed speed c.\n"
            "9. Ordinary Newtonian velocity addition applies to light.\n"
            "10. The Earth's rotation does not significantly affect the result.\n"
            "11. The apparatus is isolated from external vibrations.\n\n"
            "No directional shift was observed. How can we reconcile this with the "
            "assumptions above? Which assumptions must be wrong?"
        ),
    },

    "gr_main_elevator_light": {
        "noisy": (
            "Consider the following thought experiment and observations.\n\n"
            "1. A sealed elevator far out in space accelerates upward at 9.8 m/s^2.\n"
            "2. An observer inside feels pressed to the floor exactly as on Earth.\n"
            "3. Dropped objects fall to the floor as in ordinary gravity.\n"
            "4. This suggests acceleration and gravity are locally equivalent.\n\n"
            "Now consider light:\n"
            "5. A beam of light enters one side of the accelerating elevator.\n"
            "6. While the light crosses, the elevator continues accelerating.\n"
            "7. The observer sees the light beam curve downward.\n\n"
            "Assumptions under consideration:\n"
            "1. Acceleration and gravity produce identical local effects.\n"
            "2. Light travels in straight lines in empty space.\n"
            "3. The elevator walls are opaque.\n"
            "4. Gravity is a force that acts on mass via F = mg.\n"
            "5. Light has no mass.\n"
            "6. The elevator is small enough that tidal effects are negligible.\n"
            "7. The speed of light is much greater than the elevator's speed.\n"
            "8. Air resistance inside the elevator is negligible.\n"
            "9. The observer is at rest relative to the elevator floor.\n"
            "10. The elevator's acceleration is uniform throughout its interior.\n\n"
            "Light bends in the accelerating frame. If acceleration mimics gravity, "
            "light should bend in a gravitational field. But light has no mass.\n\n"
            "Which assumption is wrong? How can gravity affect massless light?"
        ),
    },

    "gr_free_fall_equivalence": {
        "noisy": (
            "A person falling freely from a roof does not feel their own weight.\n"
            "In a small cabin falling freely, nearby released objects float beside them.\n"
            "A person in an accelerating cabin in empty space feels weight and sees "
            "objects fall to the floor.\n\n"
            "Assumptions under consideration:\n"
            "1. Gravity is a force that pulls objects toward massive bodies.\n"
            "2. The person's mass does not change during the fall.\n"
            "3. Air resistance is negligible.\n"
            "4. Ordinary forces cannot be removed by choosing a state of motion.\n"
            "5. The cabin is small enough that gravity is uniform inside it.\n"
            "6. The person's internal organs function normally during free fall.\n"
            "7. The falling cabin does not rotate.\n"
            "8. Acceleration produces effects identical to gravity locally.\n"
            "9. The gravitational field is produced by the Earth alone.\n"
            "10. The cabin walls are perfectly insulated.\n\n"
            "How should we think about the relationship between gravity and acceleration? "
            "Which assumptions need revision?"
        ),
    },
}


# ---------------------------------------------------------------------------
# Model loading (same as physics_eval.py)
# ---------------------------------------------------------------------------

def build_model_only(checkpoint_dir, step, device):
    model_data, _, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    if device.type in {"cpu", "mps"}:
        model_data = {k: v.float() if v.dtype == torch.bfloat16 else v for k, v in model_data.items()}
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = meta_data["model_config"]
    _patch_missing_config_keys(model_config_kwargs)
    model_config = GPTConfig(**model_config_kwargs)
    _patch_missing_keys(model_data, model_config)
    with torch.device("meta"):
        model = GPT(model_config)
    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(model_data, strict=True, assign=True)
    if device.type == "cuda":
        model.bfloat16()
    model.eval()
    return model


def download_checkpoint(repo, step, cache_dir):
    step_str = f"{step:06d}"
    local_dir = os.path.join(cache_dir, f"v11-s{step}")
    if os.path.exists(os.path.join(local_dir, f"model_{step_str}.pt")):
        print(f"  [cached] step {step}")
        return local_dir
    print(f"  Downloading {repo} step={step} ...")
    snapshot_download(
        repo_id=repo,
        allow_patterns=[f"model_{step_str}.pt", f"meta_{step_str}.json", "tokenizer/*"],
        local_dir=local_dir,
    )
    return local_dir


def generate_one(engine, tokenizer, prompt_text, num_samples=1, max_tokens=1024, temperature=0.7, top_k=50):
    conversation = {
        "messages": [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": ""},
        ]
    }
    tokens = tokenizer.render_for_completion(conversation)
    prompt_len = len(tokens)
    all_tokens, _ = engine.generate_batch(tokens, num_samples=num_samples, max_tokens=max_tokens, temperature=temperature, top_k=top_k)
    completions = []
    for seq in all_tokens:
        gen_tokens = seq[prompt_len:]
        completions.append(tokenizer.decode(gen_tokens))
    return completions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="/opt/dlami/nvme/hf_cache")
    parser.add_argument("--output-dir", default="results/physics_eval_v11_rephrase")
    parser.add_argument("--step", type=int, default=630)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    repo = "mhla/gpt1900-d34-contradiction-rl-v11"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load eval config for original prompts
    eval_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "EVAL.json")
    with open(eval_path) as f:
        eval_config = json.load(f)
    original_prompts = {t["id"]: "\n".join(t["prompt_lines"]) for t in eval_config["tasks"]}

    # Download and load model once
    print(f"Downloading checkpoint step={args.step}...")
    local_dir = download_checkpoint(repo, args.step, args.cache_dir)
    tok_dir = os.path.join(local_dir, "tokenizer")
    tokenizer = RustBPETokenizer.from_directory(tok_dir)
    print(f"Loading model...")
    model = build_model_only(local_dir, args.step, device)
    engine = Engine(model, tokenizer)
    print(f"Ready. Generating with temp={args.temperature}, top_k={args.top_k}, samples={args.num_samples}\n")

    results = {}
    task_ids = list(REPHRASED_PROMPTS.keys())

    for task_id in task_ids:
        print(f"=== {task_id} ===")
        results[task_id] = {}

        # Original prompt
        prompt = original_prompts[task_id]
        completions = generate_one(engine, tokenizer, prompt, num_samples=args.num_samples, temperature=args.temperature, top_k=args.top_k)
        results[task_id]["original"] = {"prompt": prompt, "completions": completions}
        print(f"  original: {sum(len(c) for c in completions) / len(completions):.0f} avg chars")

        # Rephrased variants
        for variant_name, prompt in REPHRASED_PROMPTS[task_id].items():
            completions = generate_one(engine, tokenizer, prompt, num_samples=args.num_samples, temperature=args.temperature, top_k=args.top_k)
            results[task_id][variant_name] = {"prompt": prompt, "completions": completions}
            print(f"  {variant_name}: {sum(len(c) for c in completions) / len(completions):.0f} avg chars")

    # Save raw results
    results_path = os.path.join(args.output_dir, "generations.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {results_path}")

    # Build readable report
    report_lines = []
    for task_id in task_ids:
        report_lines.append(f"\n{'='*80}")
        report_lines.append(f"TASK: {task_id}")
        report_lines.append(f"{'='*80}")
        for variant_name, data in results[task_id].items():
            report_lines.append(f"\n--- Variant: {variant_name} ---")
            report_lines.append(f"PROMPT: {data['prompt'][:300]}...")
            for i, comp in enumerate(data["completions"]):
                report_lines.append(f"\n[Response {i+1}] ({len(comp)} chars):")
                report_lines.append(comp)
            report_lines.append("")

    report_path = os.path.join(args.output_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Saved report to {report_path}")

    del model, engine
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
