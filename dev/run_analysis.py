"""Run base model analysis - extracted from base_model_analysis.ipynb"""
import os, sys, json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from huggingface_hub import snapshot_download
from nanochat.checkpoint_manager import load_checkpoint, _patch_missing_config_keys, _patch_missing_keys
from nanochat.gpt import GPT, GPTConfig, Block
from nanochat.tokenizer import RustBPETokenizer

matplotlib.rcParams['figure.dpi'] = 120
matplotlib.rcParams['figure.facecolor'] = 'white'

FIGDIR = os.path.join(os.path.dirname(__file__), '..', 'figures', 'analysis')
os.makedirs(FIGDIR, exist_ok=True)

# ---- Load model ----
print("=== Loading model ===")
REPO_ID = "mhla/gpt1900-d34-22btok"
STEP = 10507
CACHE_DIR = os.path.expanduser("~/hf_cache")
local_dir = os.path.join(CACHE_DIR, "d34-22btok")
step_str = f"{STEP:06d}"

if not os.path.exists(os.path.join(local_dir, f"model_{step_str}.pt")):
    print(f"Downloading {REPO_ID} step={STEP} ...")
    snapshot_download(repo_id=REPO_ID, allow_patterns=[f"model_{step_str}.pt", f"meta_{step_str}.json", "tokenizer/*"], local_dir=local_dir)
else:
    print(f"[cached] {local_dir}")

tok_dir = os.path.join(local_dir, "tokenizer")
tokenizer = RustBPETokenizer.from_directory(tok_dir)
print(f"Tokenizer: vocab_size={tokenizer.get_vocab_size()}")

device = torch.device("cuda")
model_data, _, meta_data = load_checkpoint(local_dir, STEP, device, load_optimizer=False)
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
model.bfloat16()
model.eval()
print(f"Model: {model_config.n_layer} layers, {model_config.n_embd} dim, {model_config.vocab_size} vocab")
del model_data

# ---- Load eval prompts ----
with open(os.path.join(os.path.dirname(__file__), '..', 'EVAL.json')) as f:
    eval_data = json.load(f)
tasks = eval_data["tasks"]
task_names = [t["title"] for t in tasks]
prompts = ["\n".join(t["prompt_lines"]) for t in tasks]
print(f"Loaded {len(tasks)} physics eval tasks")

# ---- Analysis 1: Word pair logit probing ----
print("\n=== Analysis 1: Word Pair Logit Probing ===")

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

print("Word tokenizations:")
word_token_ids = {}
for w1, w2 in WORD_PAIRS:
    for w in (w1, w2):
        toks = tokenizer.encode(f" {w}")
        word_token_ids[w] = toks
        decoded = [tokenizer.decode([t]) for t in toks]
        marker = "" if len(toks) == 1 else f"  (multi-token: {len(toks)})"
        print(f"  '{w}' -> {toks} = {decoded}{marker}")

@torch.no_grad()
def get_logits(model, tokenizer, text):
    bos = tokenizer.get_bos_token_id()
    tokens = tokenizer.encode(text, prepend=bos)
    ids = torch.tensor([tokens], dtype=torch.long, device=model.get_device())
    logits = model(ids)
    return logits[0], tokens

# Collect last-token logits
all_logits = []
for i, prompt in enumerate(prompts):
    logits, tokens = get_logits(model, tokenizer, prompt)
    last_logits = logits[-1]
    all_logits.append(last_logits.cpu())
    probs = torch.softmax(last_logits, dim=-1)
    top5 = torch.topk(probs, 5)
    top5_str = "  ".join(f"'{tokenizer.decode([idx.item()])}' ({p:.3f})" for p, idx in zip(top5.values, top5.indices))
    print(f"  [{i}] {task_names[i]}: {len(tokens)} tok, top-5: {top5_str}")

# Build log-ratio matrix
n_prompts = len(prompts)
n_pairs = len(WORD_PAIRS)
log_ratios = np.zeros((n_prompts, n_pairs))
classical_logprobs = np.zeros((n_prompts, n_pairs))
modern_logprobs = np.zeros((n_prompts, n_pairs))

for i, logits_i in enumerate(all_logits):
    log_probs = torch.log_softmax(logits_i, dim=-1)
    for j, (w_c, w_m) in enumerate(WORD_PAIRS):
        tid_c = word_token_ids[w_c][0]
        tid_m = word_token_ids[w_m][0]
        classical_logprobs[i, j] = log_probs[tid_c].item()
        modern_logprobs[i, j] = log_probs[tid_m].item()
        log_ratios[i, j] = log_probs[tid_m].item() - log_probs[tid_c].item()

# Plot 1: Heatmap
pair_labels = [f"{w1} vs {w2}" for w1, w2 in WORD_PAIRS]
fig, ax = plt.subplots(figsize=(14, 6))
vmax = max(abs(log_ratios.min()), abs(log_ratios.max()))
im = ax.imshow(log_ratios, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
ax.set_xticks(range(n_pairs))
ax.set_xticklabels(pair_labels, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(n_prompts))
ax.set_yticklabels(task_names, fontsize=9)
ax.set_xlabel("Word Pair (classical vs modern)")
ax.set_title("log(P_modern / P_classical) — positive (red) = model leans modern")
cb = fig.colorbar(im, ax=ax, shrink=0.8)
cb.set_label("log probability ratio")
for i in range(n_prompts):
    for j in range(n_pairs):
        val = log_ratios[i, j]
        color = 'white' if abs(val) > vmax * 0.6 else 'black'
        ax.text(j, i, f"{val:.1f}", ha='center', va='center', fontsize=7, color=color)
plt.tight_layout()
fig.savefig(os.path.join(FIGDIR, '01_logit_ratio_heatmap.png'), bbox_inches='tight')
print(f"Saved 01_logit_ratio_heatmap.png")
plt.close()

# Plot 2: Raw log-probs side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
im1 = ax1.imshow(classical_logprobs, cmap='viridis', aspect='auto')
ax1.set_xticks(range(n_pairs))
ax1.set_xticklabels([w1 for w1, _ in WORD_PAIRS], rotation=45, ha='right', fontsize=9)
ax1.set_yticks(range(n_prompts))
ax1.set_yticklabels(task_names, fontsize=9)
ax1.set_title("log P(classical word)")
fig.colorbar(im1, ax=ax1, shrink=0.8)
for i in range(n_prompts):
    for j in range(n_pairs):
        ax1.text(j, i, f"{classical_logprobs[i,j]:.1f}", ha='center', va='center', fontsize=6, color='white')
im2 = ax2.imshow(modern_logprobs, cmap='viridis', aspect='auto')
ax2.set_xticks(range(n_pairs))
ax2.set_xticklabels([w2 for _, w2 in WORD_PAIRS], rotation=45, ha='right', fontsize=9)
ax2.set_yticks(range(n_prompts))
ax2.set_yticklabels(task_names, fontsize=9)
ax2.set_title("log P(modern word)")
fig.colorbar(im2, ax=ax2, shrink=0.8)
for i in range(n_prompts):
    for j in range(n_pairs):
        ax2.text(j, i, f"{modern_logprobs[i,j]:.1f}", ha='center', va='center', fontsize=6, color='white')
plt.tight_layout()
fig.savefig(os.path.join(FIGDIR, '02_raw_logprobs.png'), bbox_inches='tight')
print(f"Saved 02_raw_logprobs.png")
plt.close()

# Plot 3: Position evolution
selected = [
    (0, 0, "UV Catastrophe: continuous vs discrete"),
    (1, 1, "Photoelectric: wave vs particle"),
    (5, 2, "Michelson-Morley: aether vs vacuum"),
]
fig, axes = plt.subplots(len(selected), 1, figsize=(16, 3.5 * len(selected)))
for ax_idx, (prompt_idx, pair_idx, title) in enumerate(selected):
    ax = axes[ax_idx]
    w_classical, w_modern = WORD_PAIRS[pair_idx]
    tid_c = word_token_ids[w_classical][0]
    tid_m = word_token_ids[w_modern][0]
    logits_full, tokens = get_logits(model, tokenizer, prompts[prompt_idx])
    log_probs_full = torch.log_softmax(logits_full.cpu(), dim=-1)
    positions = np.arange(len(tokens))
    ratios = (log_probs_full[:, tid_m] - log_probs_full[:, tid_c]).numpy()
    ax.plot(positions, ratios, linewidth=1, alpha=0.8)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.fill_between(positions, ratios, 0, alpha=0.15, color='red' if ratios.mean() > 0 else 'blue')
    ax.set_ylabel(f"log(P({w_modern})/P({w_classical}))")
    ax.set_title(title)
    ax.set_xlabel("Token position")
    prompt_text = prompts[prompt_idx]
    for keyword, kcolor in [("Classical assumptions", "orange"), ("contradiction", "red"),
                            ("Which assumption", "green"), ("How can we", "green"),
                            ("How should we", "green"), ("How can we reconcile", "green")]:
        idx_in_text = prompt_text.lower().find(keyword.lower())
        if idx_in_text >= 0:
            prefix_toks = tokenizer.encode(prompt_text[:idx_in_text])
            tok_pos = len(prefix_toks)
            if tok_pos < len(tokens):
                ax.axvline(x=tok_pos, color=kcolor, linestyle=':', linewidth=1, alpha=0.7)
                ax.text(tok_pos + 1, ax.get_ylim()[1] * 0.85, keyword, fontsize=7, rotation=90, va='top', color=kcolor)
plt.tight_layout()
fig.savefig(os.path.join(FIGDIR, '03_position_evolution.png'), bbox_inches='tight')
print(f"Saved 03_position_evolution.png")
plt.close()

# ---- Analysis 2: UMAP of hidden states ----
print("\n=== Analysis 2: UMAP of Hidden States ===")

KNOWLEDGE_PROMPTS = {
    "Sciences": [
        ("Chemistry", "The modern science of chemistry arose from the older practice of alchemy, and Lavoisier demonstrated that combustion involves combination with oxygen rather than release of phlogiston."),
        ("Elements", "Mendeleev arranged the known elements into a periodic table according to their atomic weights, and predicted the properties of elements not yet discovered."),
        ("Biology", "Darwin proposed that species are not fixed but change over time through a process of natural selection acting on heritable variation."),
        ("Taxonomy", "Linnaeus devised a system of classification for all living organisms, grouping them by kingdom, class, order, genus, and species."),
        ("Astronomy", "Kepler showed that the planets move in elliptical orbits around the sun, sweeping out equal areas in equal times."),
        ("Geology", "Lyell argued that the present is the key to the past, and that the same geological forces acting today shaped the ancient earth over vast periods."),
        ("Medicine", "Pasteur and Koch established that specific diseases are caused by specific microscopic organisms, overturning the older theory of spontaneous generation."),
        ("Vaccination", "Jenner demonstrated that inoculation with cowpox matter could protect a person from the far more dangerous smallpox."),
        ("Thermodynamics", "The second law of thermodynamics states that heat flows spontaneously from hot to cold, and that no engine can convert heat entirely into work."),
        ("Electromagnetism", "Maxwell unified electricity and magnetism into a single theory and showed that light is an electromagnetic wave propagating through space."),
        ("Optics", "Newton demonstrated with a prism that white light is composed of rays of different refrangibility, each producing a distinct colour."),
        ("Mechanics", "Newton's laws of motion state that a body at rest remains at rest, force equals mass times acceleration, and every action has an equal and opposite reaction."),
    ],
    "Humanities": [
        ("Empiricism", "Locke held that the mind at birth is a blank slate, and that all knowledge is derived from experience through sensation and reflection."),
        ("Rationalism", "Descartes doubted everything that could possibly be doubted, arriving at the single certainty that the thinking self must exist."),
        ("Utilitarianism", "Bentham proposed that the rightness of an action is determined solely by the amount of happiness it produces for the greatest number."),
        ("Kant", "Kant argued that certain truths are known prior to experience, and that morality consists in acting according to a universal law one could will for all."),
        ("Theology", "Paley argued from the complexity of living things to the existence of an intelligent designer, comparing nature to a watch found upon a heath."),
        ("Biblical criticism", "Scholars applied the methods of historical and literary analysis to the scriptures, questioning traditional attributions of authorship."),
        ("Romantic poetry", "Wordsworth and Coleridge published the Lyrical Ballads, declaring that poetry should use the language of common speech and draw upon nature."),
        ("Victorian novel", "Dickens portrayed the conditions of the poor in the industrial cities, using serialised fiction to reach a wide reading public."),
        ("Classical rhetoric", "Cicero held that the ideal orator must combine wisdom and eloquence, serving the republic through the art of persuasion."),
        ("History", "Gibbon traced the decline and fall of the Roman Empire from the age of the Antonines to the fall of Constantinople."),
        ("Ancient civilisations", "The decipherment of the Rosetta Stone allowed scholars to read Egyptian hieroglyphics for the first time in over a thousand years."),
        ("Jurisprudence", "Blackstone systematised the common law of England, arguing that law derives its authority from long custom and the consent of the governed."),
    ],
    "Daily Life & Technology": [
        ("Agriculture", "The Norfolk four-course rotation of wheat, turnips, barley, and clover restored the fertility of the soil without leaving fields fallow."),
        ("Animal husbandry", "Bakewell improved the breeding of cattle and sheep by carefully selecting animals for desirable traits over many generations."),
        ("Steam engine", "Watt improved the steam engine by adding a separate condenser, greatly increasing its efficiency and making it practical for driving machinery."),
        ("Textile mills", "The spinning jenny and the power loom transformed the production of cloth from a cottage industry into a factory system."),
        ("Railways", "Stephenson's locomotive demonstrated that steam-powered railways could carry passengers and goods faster and more cheaply than horse-drawn coaches."),
        ("Telegraph", "Morse's electric telegraph allowed messages to be sent instantaneously over great distances along a wire, revolutionising communication."),
        ("Navigation", "Harrison's marine chronometer solved the longitude problem, enabling navigators to determine their position at sea with great accuracy."),
        ("Cartography", "The triangulation surveys of the Ordnance Survey produced accurate maps of the British Isles based on precise measurement."),
        ("Banking", "The Bank of England was established to manage the national debt and provide a stable currency backed by gold reserves."),
        ("Trade", "The East India Company held a monopoly on trade with the East, importing tea, silk, and spices in exchange for silver and manufactured goods."),
        ("Bridges", "Brunel designed iron bridges and steamships of unprecedented scale, demonstrating the new possibilities of engineering with metal."),
        ("Printing", "The steam-powered printing press made books and newspapers cheap enough for the common reader, vastly expanding the literate public."),
    ],
    "Geography & Culture": [
        ("Europe", "The Congress of Vienna redrawn the map of Europe after the defeat of Napoleon, establishing a balance of power among the great nations."),
        ("Americas", "The Monroe Doctrine declared that the American continents were no longer open to European colonisation or interference."),
        ("Asia", "The opening of Japan by Commodore Perry ended two centuries of isolation and began the rapid modernisation of the country."),
        ("Africa", "Livingstone explored the interior of Africa, mapping the course of the Zambezi and bringing reports of the peoples he encountered."),
        ("Oceania", "Cook charted the coasts of New Zealand and eastern Australia, claiming new territories for the British Crown."),
        ("British Empire", "At the height of its power the British Empire governed territories on every continent, from India to Canada to the Cape Colony."),
        ("Ottoman Empire", "The Ottoman Empire controlled a vast territory stretching from the Balkans through Anatolia to the Arabian Peninsula and North Africa."),
        ("Qing dynasty", "The Qing dynasty ruled China for nearly three centuries, maintaining the examination system and expanding the empire to its greatest extent."),
        ("Music", "Beethoven expanded the symphony beyond anything previously imagined, expressing profound emotion through purely instrumental music."),
        ("Architecture", "The Gothic Revival swept through Europe, with architects imitating the pointed arches and flying buttresses of medieval cathedrals."),
        ("Linguistics", "Jones observed that Sanskrit, Greek, and Latin share a common ancestor, laying the foundation for comparative philology."),
        ("Ethnography", "Travellers and missionaries compiled vocabularies and grammars of the languages of Africa, the Pacific, and the Americas."),
    ],
}

# Register hooks
hidden_states = {}
def make_hook(layer_idx):
    def hook_fn(module, input, output):
        hidden_states[layer_idx] = output.detach()
    return hook_fn

hooks = []
for i, block in enumerate(model.transformer.h):
    hooks.append(block.register_forward_hook(make_hook(i)))
print(f"Registered {len(hooks)} hooks")

@torch.no_grad()
def get_hidden_states(model, tokenizer, text, layer_indices):
    bos = tokenizer.get_bos_token_id()
    tokens = tokenizer.encode(text, prepend=bos)
    ids = torch.tensor([tokens], dtype=torch.long, device=model.get_device())
    _ = model(ids)
    return {li: hidden_states[li][0, -1, :].float().cpu().numpy() for li in layer_indices}

n_layers = model_config.n_layer
analysis_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
print(f"Collecting hidden states at layers: {analysis_layers}")

all_labels, all_categories, all_texts = [], [], []
all_hidden = {li: [] for li in analysis_layers}

for cat, items in KNOWLEDGE_PROMPTS.items():
    for label, text in items:
        all_labels.append(label)
        all_categories.append(cat)
        all_texts.append(text)
        h = get_hidden_states(model, tokenizer, text, analysis_layers)
        for li in analysis_layers:
            all_hidden[li].append(h[li])

for li in analysis_layers:
    all_hidden[li] = np.stack(all_hidden[li])
n_total = len(all_labels)
print(f"Collected {n_total} vectors, shape={all_hidden[analysis_layers[0]].shape}")

# UMAP + PCA
# Mean-center hidden states to remove the dominant shared component
# (transformers have a strong mean direction that makes all cosine sims ~1.0)
last_layer = analysis_layers[-1]
X = all_hidden[last_layer]
X_centered = X - X.mean(axis=0, keepdims=True)
X_norm = X_centered / (np.linalg.norm(X_centered, axis=1, keepdims=True) + 1e-8)

reducer = umap.UMAP(n_neighbors=12, min_dist=0.3, metric='euclidean', random_state=42)
umap_2d = reducer.fit_transform(X_norm)
pca = PCA(n_components=2)
pca_2d = pca.fit_transform(X_norm)
print(f"PCA explained variance: {pca.explained_variance_ratio_[0]:.3f}, {pca.explained_variance_ratio_[1]:.3f}")

cat_colors = {"Sciences": "#e74c3c", "Humanities": "#3498db", "Daily Life & Technology": "#2ecc71", "Geography & Culture": "#f39c12"}
cat_markers = {"Sciences": "o", "Humanities": "s", "Daily Life & Technology": "D", "Geography & Culture": "^"}

def plot_scatter(ax, coords_2d, title):
    for cat in cat_colors:
        idxs = [i for i, c in enumerate(all_categories) if c == cat]
        ax.scatter(coords_2d[idxs, 0], coords_2d[idxs, 1], c=cat_colors[cat], marker=cat_markers[cat],
                  label=cat, s=60, alpha=0.8, edgecolors='white', linewidth=0.5)
        for idx in idxs:
            ax.annotate(all_labels[idx], (coords_2d[idx, 0], coords_2d[idx, 1]),
                       fontsize=6, alpha=0.7, ha='center', va='bottom', xytext=(0, 4), textcoords='offset points')
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, loc='best')
    ax.set_xticks([]); ax.set_yticks([])

# Plot 4: UMAP + PCA side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
plot_scatter(ax1, umap_2d, f"UMAP — Layer {last_layer} (final) Hidden States")
plot_scatter(ax2, pca_2d, f"PCA — Layer {last_layer} (final) Hidden States")
plt.tight_layout()
fig.savefig(os.path.join(FIGDIR, '04_umap_pca.png'), bbox_inches='tight')
print(f"Saved 04_umap_pca.png")
plt.close()

# Plot 5: Multi-layer UMAP
fig, axes = plt.subplots(1, len(analysis_layers), figsize=(5 * len(analysis_layers), 5))
for ax_idx, li in enumerate(analysis_layers):
    ax = axes[ax_idx]
    X_li = all_hidden[li]
    X_li_centered = X_li - X_li.mean(axis=0, keepdims=True)
    X_li_norm = X_li_centered / (np.linalg.norm(X_li_centered, axis=1, keepdims=True) + 1e-8)
    reducer_li = umap.UMAP(n_neighbors=12, min_dist=0.3, metric='euclidean', random_state=42)
    umap_li = reducer_li.fit_transform(X_li_norm)
    for cat in cat_colors:
        idxs = [i for i, c in enumerate(all_categories) if c == cat]
        ax.scatter(umap_li[idxs, 0], umap_li[idxs, 1], c=cat_colors[cat], marker=cat_markers[cat],
                  label=cat if ax_idx == 0 else None, s=40, alpha=0.8, edgecolors='white', linewidth=0.3)
    ax.set_title(f"Layer {li}", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
axes[0].legend(fontsize=7, loc='best')
fig.suptitle("UMAP across layers: how knowledge organization emerges through depth", fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIGDIR, '05_multi_layer_umap.png'), bbox_inches='tight')
print(f"Saved 05_multi_layer_umap.png")
plt.close()

# Plot 6: Cosine similarity heatmap (mean-centered)
X_final_centered = X - X.mean(axis=0, keepdims=True)
X_final_norm = X_final_centered / (np.linalg.norm(X_final_centered, axis=1, keepdims=True) + 1e-8)
cos_sim = X_final_norm @ X_final_norm.T
cat_order = ["Sciences", "Humanities", "Daily Life & Technology", "Geography & Culture"]
sorted_idxs = []
for cat in cat_order:
    sorted_idxs.extend([i for i, c in enumerate(all_categories) if c == cat])
cos_sim_sorted = cos_sim[np.ix_(sorted_idxs, sorted_idxs)]
sorted_labels = [all_labels[i] for i in sorted_idxs]

fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(cos_sim_sorted, cmap='RdYlBu_r', vmin=-0.2, vmax=1.0)
ax.set_xticks(range(n_total))
ax.set_xticklabels(sorted_labels, rotation=90, fontsize=6)
ax.set_yticks(range(n_total))
ax.set_yticklabels(sorted_labels, fontsize=6)
cumsum = 0
for cat in cat_order:
    count = sum(1 for c in all_categories if c == cat)
    ax.axhline(y=cumsum - 0.5, color='black', linewidth=1)
    ax.axvline(x=cumsum - 0.5, color='black', linewidth=1)
    mid = cumsum + count / 2
    ax.text(-2, mid, cat, fontsize=8, ha='right', va='center', fontweight='bold', color=cat_colors[cat])
    cumsum += count
ax.axhline(y=cumsum - 0.5, color='black', linewidth=1)
ax.axvline(x=cumsum - 0.5, color='black', linewidth=1)
cb = fig.colorbar(im, ax=ax, shrink=0.8)
cb.set_label("Cosine similarity")
ax.set_title(f"Pairwise cosine similarity of hidden states (layer {last_layer})", fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(FIGDIR, '06_cosine_similarity.png'), bbox_inches='tight')
print(f"Saved 06_cosine_similarity.png")
plt.close()

# Cleanup
for h in hooks:
    h.remove()

print(f"\n=== Done! All figures saved to {FIGDIR} ===")
