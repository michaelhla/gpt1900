"""Targeted logit probing: one completion prompt + word pair per physics eval task.
Compares base model vs physics SFT expanded."""
import os, sys, json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from nanochat.checkpoint_manager import load_checkpoint, _patch_missing_config_keys, _patch_missing_keys
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import RustBPETokenizer

matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['figure.facecolor'] = 'white'

FIGDIR = os.path.join(os.path.dirname(__file__), '..', 'figures', 'analysis')
os.makedirs(FIGDIR, exist_ok=True)

# ---- Model loading ----
def load_model(checkpoint_dir, step, device):
    model_data, _, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    cfg = meta_data["model_config"]
    _patch_missing_config_keys(cfg)
    model_config = GPTConfig(**cfg)
    _patch_missing_keys(model_data, model_config)
    with torch.device("meta"):
        model = GPT(model_config)
    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(model_data, strict=True, assign=True)
    model.bfloat16()
    model.eval()
    return model, model_config

device = torch.device("cuda")

print("=== Loading models ===")
base_dir = os.path.expanduser("~/hf_cache/d34-22btok")
base_model, _ = load_model(base_dir, 10507, device)
tokenizer = RustBPETokenizer.from_directory(os.path.join(base_dir, "tokenizer"))

phys_dir = "/opt/dlami/nvme/hf_cache/d34-physicssft-expanded"
phys_model, _ = load_model(phys_dir, 14400, device)
print("Models loaded.")

# ---- Targeted prompts: one per physics eval task ----
# Each entry: (task_short_name, completion_prompt, classical_word, modern_word)
PROBES = [
    ("UV Catastrophe",
     "The energy of each mode of vibration in a heated cavity must be",
     "continuous", "discrete"),

    ("Photoelectric\nEffect",
     "The experimental evidence suggests that light is best described as a",
     "wave", "particle"),

    ("Chasing a\nLight Beam",
     "An observer moving alongside a beam of light would find its speed to be",
     "reduced", "constant"),

    ("Approaching c",
     "As an object approaches the speed of light, its mass effectively becomes",
     "unchanged", "infinite"),

    ("Train &\nLightning",
     "Two events that are simultaneous in one frame of reference are in another frame",
     "simultaneous", "separated"),

    ("Michelson-\nMorley",
     "The experiment showed that light does not require a medium such as the",
     "aether", "vacuum"),

    ("Elevator\n& Light",
     "If acceleration and gravity are equivalent, then gravity must affect not just mass but also",
     "matter", "light"),

    ("Free Fall &\nAcceleration",
     "The fact that gravity can be eliminated by free fall suggests it is not a force but a property of",
     "matter", "space"),
]

# ---- Get logprobs for each probe ----
@torch.no_grad()
def get_word_logprobs(model, tokenizer, prompt, words):
    """Return log-probabilities for each word as the next token after prompt."""
    bos = tokenizer.get_bos_token_id()
    tokens = tokenizer.encode(prompt, prepend=bos)
    ids = torch.tensor([tokens], dtype=torch.long, device=model.get_device())
    logits = model(ids)
    log_probs = torch.log_softmax(logits[0, -1, :], dim=-1)
    probs = torch.softmax(logits[0, -1, :], dim=-1)
    result = {}
    for w in words:
        toks = tokenizer.encode(f" {w}")
        tid = toks[0]  # first token
        result[w] = {
            'logprob': log_probs[tid].item(),
            'prob': probs[tid].item(),
        }
    return result

print("\n=== Computing logprobs ===")
results = []
for task_name, prompt, w_classical, w_modern in PROBES:
    base_lp = get_word_logprobs(base_model, tokenizer, prompt, [w_classical, w_modern])
    phys_lp = get_word_logprobs(phys_model, tokenizer, prompt, [w_classical, w_modern])
    results.append({
        'task': task_name,
        'prompt': prompt,
        'classical': w_classical,
        'modern': w_modern,
        'base_classical_prob': base_lp[w_classical]['prob'],
        'base_modern_prob': base_lp[w_modern]['prob'],
        'phys_classical_prob': phys_lp[w_classical]['prob'],
        'phys_modern_prob': phys_lp[w_modern]['prob'],
    })
    print(f"  {task_name.replace(chr(10), ' ')}: "
          f"base P({w_classical})={base_lp[w_classical]['prob']:.4f} P({w_modern})={base_lp[w_modern]['prob']:.4f} | "
          f"phys P({w_classical})={phys_lp[w_classical]['prob']:.4f} P({w_modern})={phys_lp[w_modern]['prob']:.4f}")

# ---- Plot: grouped bar chart ----
n = len(results)
task_labels = [r['task'] for r in results]
x = np.arange(n)
bar_width = 0.18

fig, ax = plt.subplots(figsize=(16, 7))

# 4 bars per task: base_classical, base_modern, phys_classical, phys_modern
base_classical = [r['base_classical_prob'] for r in results]
base_modern = [r['base_modern_prob'] for r in results]
phys_classical = [r['phys_classical_prob'] for r in results]
phys_modern = [r['phys_modern_prob'] for r in results]

b1 = ax.bar(x - 1.5*bar_width, base_classical, bar_width, label='Base: classical word',
            color='#3498db', alpha=0.8)
b2 = ax.bar(x - 0.5*bar_width, base_modern, bar_width, label='Base: modern word',
            color='#e74c3c', alpha=0.8)
b3 = ax.bar(x + 0.5*bar_width, phys_classical, bar_width, label='Physics SFT: classical word',
            color='#3498db', alpha=0.4, hatch='//')
b4 = ax.bar(x + 1.5*bar_width, phys_modern, bar_width, label='Physics SFT: modern word',
            color='#e74c3c', alpha=0.4, hatch='//')

# Add word labels under each group
for i, r in enumerate(results):
    y_offset = -0.003
    ax.text(i - bar_width, y_offset, r['classical'], ha='center', va='top',
           fontsize=7, color='#2980b9', fontstyle='italic')
    ax.text(i + bar_width, y_offset, r['modern'], ha='center', va='top',
           fontsize=7, color='#c0392b', fontstyle='italic')

ax.set_xticks(x)
ax.set_xticklabels(task_labels, fontsize=9, ha='center')
ax.set_ylabel("P(word | prompt)", fontsize=11)
ax.set_title("Targeted next-token probabilities: classical vs modern concept per physics task\n"
             "Base model (d34-22btok) vs Physics SFT Expanded", fontsize=13)
ax.legend(fontsize=9, loc='upper right')
ax.set_ylim(bottom=min(0, ax.get_ylim()[0] - 0.01))

# Add prompt text as annotation below
for i, r in enumerate(results):
    # Truncate prompt for display
    short_prompt = r['prompt']
    if len(short_prompt) > 60:
        short_prompt = short_prompt[-57:] + "..."
        # find a word boundary
        space = short_prompt.find(' ')
        if space > 0:
            short_prompt = "..." + short_prompt[space:]
    ax.text(i, ax.get_ylim()[0] + 0.001, f'"{short_prompt} ___"',
           ha='center', va='bottom', fontsize=4.5, color='gray', rotation=45)

plt.tight_layout()
fig.savefig(os.path.join(FIGDIR, '16_targeted_logits.png'), bbox_inches='tight')
print(f"\nSaved 16_targeted_logits.png")
plt.close()

# ---- Plot 2: log-odds ratio comparison ----
fig, ax = plt.subplots(figsize=(14, 6))

base_log_ratio = [np.log(r['base_modern_prob'] / max(r['base_classical_prob'], 1e-10)) for r in results]
phys_log_ratio = [np.log(r['phys_modern_prob'] / max(r['phys_classical_prob'], 1e-10)) for r in results]

bar_width = 0.35
b1 = ax.bar(x - bar_width/2, base_log_ratio, bar_width, label='Base model',
            color='#3498db', alpha=0.8)
b2 = ax.bar(x + bar_width/2, phys_log_ratio, bar_width, label='Physics SFT Expanded',
            color='#e74c3c', alpha=0.8)

ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(task_labels, fontsize=9)
ax.set_ylabel("log(P_modern / P_classical)", fontsize=11)
ax.set_title("Log-odds ratio: positive = model favors modern concept", fontsize=13)
ax.legend(fontsize=10)

# Annotate with word pairs
for i, r in enumerate(results):
    ax.text(i, ax.get_ylim()[0] + 0.1, f"{r['classical']}\nvs\n{r['modern']}",
           ha='center', va='bottom', fontsize=7, color='gray')

plt.tight_layout()
fig.savefig(os.path.join(FIGDIR, '17_targeted_log_odds.png'), bbox_inches='tight')
print(f"Saved 17_targeted_log_odds.png")
plt.close()

print(f"\n=== Done! ===")
