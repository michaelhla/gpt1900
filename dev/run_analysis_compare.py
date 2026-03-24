"""Compare logit probing: base d34-22btok vs d34-physicssft-expanded"""
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

matplotlib.rcParams['figure.dpi'] = 120
matplotlib.rcParams['figure.facecolor'] = 'white'

FIGDIR = os.path.join(os.path.dirname(__file__), '..', 'figures', 'analysis')
os.makedirs(FIGDIR, exist_ok=True)

# --- Model loading helper ---
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

# --- Load base model ---
print("=== Loading base model (d34-22btok) ===")
base_dir = os.path.expanduser("~/hf_cache/d34-22btok")
base_model, base_config = load_model(base_dir, 10507, device)
print(f"  {base_config.n_layer} layers, {base_config.n_embd} dim")

# Load tokenizer from base
tok_dir = os.path.join(base_dir, "tokenizer")
tokenizer = RustBPETokenizer.from_directory(tok_dir)

# --- Load physics SFT expanded model ---
print("=== Loading physics SFT expanded model ===")
phys_dir = "/opt/dlami/nvme/hf_cache/d34-physicssft-expanded"
phys_model, phys_config = load_model(phys_dir, 14400, device)
print(f"  {phys_config.n_layer} layers, {phys_config.n_embd} dim")

# --- Load eval prompts ---
with open(os.path.join(os.path.dirname(__file__), '..', 'EVAL.json')) as f:
    eval_data = json.load(f)
tasks = eval_data["tasks"]
task_names = [t["title"] for t in tasks]
prompts = ["\n".join(t["prompt_lines"]) for t in tasks]

# --- Word pairs ---
WORD_PAIRS = [
    ("continuous", "discrete"),
    ("wave", "particle"),
    ("aether", "vacuum"),
    ("absolute", "relative"),
    ("classical", "quantum"),
    ("energy", "quanta"),
    ("ether", "space"),
    ("force", "geometry"),
    ("simultaneous", "relative"),
    ("mass", "energy"),
]

word_token_ids = {}
for w1, w2 in WORD_PAIRS:
    for w in (w1, w2):
        word_token_ids[w] = tokenizer.encode(f" {w}")

@torch.no_grad()
def get_last_logits(model, tokenizer, text):
    bos = tokenizer.get_bos_token_id()
    tokens = tokenizer.encode(text, prepend=bos)
    ids = torch.tensor([tokens], dtype=torch.long, device=model.get_device())
    logits = model(ids)
    return logits[0, -1, :].cpu()  # last token logits

@torch.no_grad()
def get_all_logits(model, tokenizer, text):
    bos = tokenizer.get_bos_token_id()
    tokens = tokenizer.encode(text, prepend=bos)
    ids = torch.tensor([tokens], dtype=torch.long, device=model.get_device())
    logits = model(ids)
    return logits[0].cpu(), tokens

def compute_log_ratios(model, tokenizer, prompts):
    n_prompts = len(prompts)
    n_pairs = len(WORD_PAIRS)
    log_ratios = np.zeros((n_prompts, n_pairs))
    classical_lp = np.zeros((n_prompts, n_pairs))
    modern_lp = np.zeros((n_prompts, n_pairs))
    for i, prompt in enumerate(prompts):
        logits = get_last_logits(model, tokenizer, prompt)
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        top5 = torch.topk(probs, 5)
        top5_str = "  ".join(f"'{tokenizer.decode([idx.item()])}' ({p:.3f})" for p, idx in zip(top5.values, top5.indices))
        print(f"    [{i}] {task_names[i]}: top-5: {top5_str}")
        for j, (wc, wm) in enumerate(WORD_PAIRS):
            tc = word_token_ids[wc][0]
            tm = word_token_ids[wm][0]
            classical_lp[i, j] = log_probs[tc].item()
            modern_lp[i, j] = log_probs[tm].item()
            log_ratios[i, j] = log_probs[tm].item() - log_probs[tc].item()
    return log_ratios, classical_lp, modern_lp

# --- Compute for both models ---
print("\n--- Base model logits ---")
base_ratios, base_classical, base_modern = compute_log_ratios(base_model, tokenizer, prompts)

print("\n--- Physics SFT expanded logits ---")
phys_ratios, phys_classical, phys_modern = compute_log_ratios(phys_model, tokenizer, prompts)

# --- Plot 1: Side-by-side heatmaps ---
pair_labels = [f"{w1} vs {w2}" for w1, w2 in WORD_PAIRS]
n_prompts = len(prompts)
n_pairs = len(WORD_PAIRS)

# Use shared color scale
all_vals = np.concatenate([base_ratios.ravel(), phys_ratios.ravel()])
vmax = max(abs(all_vals.min()), abs(all_vals.max()))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 7), sharey=True)

for ax, ratios, title in [(ax1, base_ratios, "Base model (d34-22btok)"),
                           (ax2, phys_ratios, "Physics SFT Expanded (d34-physicssft-expanded)")]:
    im = ax.imshow(ratios, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n_pairs))
    ax.set_xticklabels(pair_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n_prompts))
    ax.set_yticklabels(task_names, fontsize=9)
    ax.set_title(title, fontsize=12)
    for i in range(n_prompts):
        for j in range(n_pairs):
            val = ratios[i, j]
            color = 'white' if abs(val) > vmax * 0.6 else 'black'
            ax.text(j, i, f"{val:.1f}", ha='center', va='center', fontsize=7, color=color)

fig.colorbar(im, ax=[ax1, ax2], shrink=0.8, label="log(P_modern / P_classical)")
fig.suptitle("log(P_modern / P_classical) — positive (red) = model leans modern", fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(FIGDIR, '07_compare_heatmaps.png'), bbox_inches='tight')
print(f"\nSaved 07_compare_heatmaps.png")
plt.close()

# --- Plot 2: Difference heatmap (physics SFT - base) ---
diff = phys_ratios - base_ratios
vmax_diff = max(abs(diff.min()), abs(diff.max()))

fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(diff, cmap='PiYG', aspect='auto', vmin=-vmax_diff, vmax=vmax_diff)
ax.set_xticks(range(n_pairs))
ax.set_xticklabels(pair_labels, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(n_prompts))
ax.set_yticklabels(task_names, fontsize=9)
ax.set_title("Shift from physics SFT: log-ratio(physics SFT) - log-ratio(base)\nGreen = physics SFT shifted toward modern concept", fontsize=11)
cb = fig.colorbar(im, ax=ax, shrink=0.8)
cb.set_label("Δ log probability ratio")
for i in range(n_prompts):
    for j in range(n_pairs):
        val = diff[i, j]
        color = 'white' if abs(val) > vmax_diff * 0.5 else 'black'
        ax.text(j, i, f"{val:+.1f}", ha='center', va='center', fontsize=7, color=color)
plt.tight_layout()
fig.savefig(os.path.join(FIGDIR, '08_diff_heatmap.png'), bbox_inches='tight')
print(f"Saved 08_diff_heatmap.png")
plt.close()

# --- Plot 3: Position evolution comparison for selected prompts ---
selected = [
    (0, 0, "UV Catastrophe: continuous vs discrete"),
    (1, 1, "Photoelectric: wave vs particle"),
    (5, 2, "Michelson-Morley: aether vs vacuum"),
]

fig, axes = plt.subplots(len(selected), 1, figsize=(16, 4 * len(selected)))

for ax_idx, (prompt_idx, pair_idx, title) in enumerate(selected):
    ax = axes[ax_idx]
    wc, wm = WORD_PAIRS[pair_idx]
    tc = word_token_ids[wc][0]
    tm = word_token_ids[wm][0]

    # Base model
    logits_base, tokens = get_all_logits(base_model, tokenizer, prompts[prompt_idx])
    lp_base = torch.log_softmax(logits_base, dim=-1)
    ratio_base = (lp_base[:, tm] - lp_base[:, tc]).numpy()

    # Physics SFT
    logits_phys, _ = get_all_logits(phys_model, tokenizer, prompts[prompt_idx])
    lp_phys = torch.log_softmax(logits_phys, dim=-1)
    ratio_phys = (lp_phys[:, tm] - lp_phys[:, tc]).numpy()

    positions = np.arange(len(tokens))
    ax.plot(positions, ratio_base, linewidth=1, alpha=0.7, color='#3498db', label='Base (d34-22btok)')
    ax.plot(positions, ratio_phys, linewidth=1, alpha=0.7, color='#e74c3c', label='Physics SFT expanded')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.fill_between(positions, ratio_base, ratio_phys, alpha=0.1, color='purple')
    ax.set_ylabel(f"log(P({wm})/P({wc}))")
    ax.set_title(title)
    ax.set_xlabel("Token position")
    ax.legend(fontsize=8)

    # Annotate prompt sections
    prompt_text = prompts[prompt_idx]
    for keyword, kcolor in [("Classical assumptions", "orange"), ("contradiction", "darkred"),
                            ("Which assumption", "green"), ("How can we", "green"),
                            ("How should we", "green"), ("How can we reconcile", "green")]:
        idx_in_text = prompt_text.lower().find(keyword.lower())
        if idx_in_text >= 0:
            prefix_toks = tokenizer.encode(prompt_text[:idx_in_text])
            tok_pos = len(prefix_toks)
            if tok_pos < len(tokens):
                ax.axvline(x=tok_pos, color=kcolor, linestyle=':', linewidth=1, alpha=0.5)

plt.tight_layout()
fig.savefig(os.path.join(FIGDIR, '09_position_compare.png'), bbox_inches='tight')
print(f"Saved 09_position_compare.png")
plt.close()

# --- Plot 4: Bar chart of average shift per word pair ---
mean_diff = diff.mean(axis=0)  # average across prompts
sort_idx = np.argsort(mean_diff)

fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#27ae60' if v > 0 else '#c0392b' for v in mean_diff[sort_idx]]
bars = ax.barh(range(n_pairs), mean_diff[sort_idx], color=colors, alpha=0.8)
ax.set_yticks(range(n_pairs))
ax.set_yticklabels([pair_labels[i] for i in sort_idx], fontsize=10)
ax.set_xlabel("Mean Δ log(P_modern / P_classical)")
ax.set_title("Average shift from physics SFT training\nGreen = shifted toward modern concept")
ax.axvline(x=0, color='black', linewidth=0.5)
for i, v in enumerate(mean_diff[sort_idx]):
    ax.text(v + 0.05 * np.sign(v), i, f"{v:+.2f}", va='center', fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(FIGDIR, '10_mean_shift_bars.png'), bbox_inches='tight')
print(f"Saved 10_mean_shift_bars.png")
plt.close()

print(f"\n=== Done! Comparison figures saved to {FIGDIR} ===")
