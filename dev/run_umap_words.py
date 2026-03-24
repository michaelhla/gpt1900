"""UMAP of individual word/concept embeddings from the pre-1900 base model."""
import os, sys, json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from nanochat.checkpoint_manager import load_checkpoint, _patch_missing_config_keys, _patch_missing_keys
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import RustBPETokenizer

matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['figure.facecolor'] = 'white'

FIGDIR = os.path.join(os.path.dirname(__file__), '..', 'figures', 'analysis')
os.makedirs(FIGDIR, exist_ok=True)

# ---- Load model ----
print("=== Loading model ===")
CACHE_DIR = os.path.expanduser("~/hf_cache")
local_dir = os.path.join(CACHE_DIR, "d34-22btok")
tok_dir = os.path.join(local_dir, "tokenizer")
tokenizer = RustBPETokenizer.from_directory(tok_dir)

device = torch.device("cuda")
model_data, _, meta_data = load_checkpoint(local_dir, 10507, device, load_optimizer=False)
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
print(f"Model: {model_config.n_layer}L, {model_config.n_embd}d")
del model_data

# ---- Hook for final layer hidden state ----
final_layer_idx = model_config.n_layer - 1
final_hidden = {}

def hook_fn(module, input, output):
    final_hidden['h'] = output.detach()

hook = list(model.transformer.h)[-1].register_forward_hook(hook_fn)

@torch.no_grad()
def get_word_embedding(word):
    """Get final-layer last-token hidden state for a word."""
    bos = tokenizer.get_bos_token_id()
    tokens = tokenizer.encode(f" {word}", prepend=bos)
    ids = torch.tensor([tokens], dtype=torch.long, device=device)
    _ = model(ids)
    return final_hidden['h'][0, -1, :].float().cpu().numpy()

# ---- Word lists by category ----
WORDS = {
    "Sciences": [
        # Physics & mechanics
        "gravity", "velocity", "acceleration", "momentum", "inertia", "friction",
        "pendulum", "lever", "pulley", "equilibrium", "pressure", "density",
        "mass", "weight", "force", "motion", "trajectory", "collision",
        "elasticity", "viscosity", "buoyancy", "torque", "oscillation", "resonance",
        # Optics & waves
        "light", "prism", "refraction", "reflection", "diffraction", "polarisation",
        "wavelength", "spectrum", "lens", "mirror", "telescope", "microscope",
        "colour", "brightness", "shadow", "luminous", "phosphorescent", "radiant",
        # Electricity & magnetism
        "electricity", "magnetism", "current", "voltage", "resistance", "capacitor",
        "battery", "telegraph", "conductor", "insulator", "galvanic", "electrolysis",
        "compass", "lodestone", "induction", "dynamo", "spark", "lightning",
        # Thermodynamics
        "heat", "temperature", "entropy", "steam", "boiling", "freezing",
        "combustion", "caloric", "conduction", "convection", "radiation", "thermometer",
        # Chemistry
        "oxygen", "hydrogen", "nitrogen", "carbon", "sulphur", "phosphorus",
        "acid", "alkali", "salt", "compound", "element", "atom",
        "molecule", "reaction", "combustion", "distillation", "precipitate", "solution",
        "mercury", "iron", "copper", "gold", "silver", "lead",
        "chlorine", "potassium", "sodium", "arsenic", "zinc", "tin",
        # Biology & natural history
        "species", "genus", "evolution", "heredity", "variation", "adaptation",
        "fossil", "organism", "cell", "tissue", "organ", "muscle",
        "nerve", "blood", "bone", "skeleton", "embryo", "larva",
        "mammal", "reptile", "insect", "bird", "fish", "amphibian",
        "flower", "seed", "root", "leaf", "pollen", "fern",
        "bacteria", "parasite", "infection", "vaccination", "anatomy", "physiology",
        # Astronomy & geology
        "planet", "star", "comet", "eclipse", "orbit", "constellation",
        "nebula", "meteor", "asteroid", "telescope", "observatory", "parallax",
        "volcano", "earthquake", "glacier", "fossil", "stratum", "mineral",
        "granite", "limestone", "sandstone", "basalt", "quartz", "crystal",
        # Mathematics
        "algebra", "geometry", "calculus", "logarithm", "equation", "theorem",
        "integral", "derivative", "trigonometry", "probability", "infinity", "series",
    ],
    "Humanities": [
        # Philosophy
        "reason", "logic", "truth", "knowledge", "wisdom", "virtue",
        "morality", "ethics", "justice", "liberty", "equality", "duty",
        "conscience", "soul", "mind", "idea", "perception", "experience",
        "empiricism", "rationalism", "idealism", "materialism", "skepticism", "stoicism",
        "metaphysics", "epistemology", "ontology", "dialectic", "syllogism", "axiom",
        "utilitarianism", "determinism", "causation", "substance", "essence", "existence",
        # Religion & theology
        "God", "prayer", "faith", "scripture", "revelation", "salvation",
        "sin", "redemption", "providence", "miracle", "prophecy", "covenant",
        "church", "cathedral", "bishop", "priest", "monastery", "pilgrimage",
        "sermon", "hymn", "psalm", "gospel", "apostle", "martyr",
        "baptism", "communion", "confession", "penance", "sacrament", "resurrection",
        "heaven", "hell", "purgatory", "angel", "devil", "creation",
        # Literature & rhetoric
        "poetry", "sonnet", "ode", "elegy", "ballad", "epic",
        "novel", "tragedy", "comedy", "satire", "allegory", "fable",
        "metaphor", "irony", "rhetoric", "eloquence", "narrative", "verse",
        "stanza", "couplet", "rhyme", "metre", "prose", "essay",
        "author", "critic", "publisher", "library", "manuscript", "edition",
        "Shakespeare", "Milton", "Homer", "Virgil", "Dante", "Goethe",
        # History & law
        "king", "queen", "emperor", "parliament", "constitution", "republic",
        "revolution", "treaty", "alliance", "conquest", "empire", "colony",
        "feudal", "vassal", "knight", "crusade", "reformation", "renaissance",
        "democracy", "aristocracy", "tyranny", "sovereignty", "diplomacy", "treason",
        "statute", "verdict", "jury", "judge", "advocate", "precedent",
        "contract", "property", "inheritance", "testimony", "oath", "tribunal",
        # Education & scholarship
        "university", "professor", "lecture", "examination", "scholarship", "curriculum",
        "Latin", "Greek", "Hebrew", "Sanskrit", "grammar", "philology",
        "dictionary", "encyclopaedia", "atlas", "gazette", "journal", "pamphlet",
        "academy", "seminary", "tutor", "pupil", "diploma", "dissertation",
    ],
    "Daily Life & Technology": [
        # Agriculture & food
        "wheat", "barley", "oats", "corn", "rice", "potato",
        "plough", "harvest", "sowing", "irrigation", "fertiliser", "rotation",
        "cattle", "sheep", "horse", "pig", "goat", "poultry",
        "dairy", "butter", "cheese", "bread", "flour", "yeast",
        "orchard", "vineyard", "pasture", "meadow", "haystack", "granary",
        "cider", "ale", "wine", "tea", "coffee", "sugar",
        # Industry & manufacturing
        "factory", "mill", "furnace", "forge", "anvil", "hammer",
        "loom", "spindle", "cotton", "wool", "silk", "linen",
        "iron", "steel", "brass", "coal", "coke", "smelting",
        "engine", "boiler", "piston", "cylinder", "valve", "flywheel",
        "chimney", "warehouse", "workshop", "foundry", "kiln", "tannery",
        "patent", "invention", "machinery", "manufacture", "production", "labour",
        # Transport & communication
        "railway", "locomotive", "carriage", "coach", "wagon", "canal",
        "steamship", "sailing", "harbour", "wharf", "dock", "lighthouse",
        "road", "turnpike", "bridge", "tunnel", "ferry", "omnibus",
        "postage", "letter", "newspaper", "telegraph", "messenger", "dispatch",
        "bicycle", "saddle", "horseshoe", "stable", "coachman", "driver",
        "anchor", "rudder", "mast", "sail", "cargo", "ballast",
        # Household & domestic
        "candle", "lamp", "fireplace", "stove", "oven", "kettle",
        "soap", "linen", "curtain", "carpet", "furniture", "wardrobe",
        "clock", "watch", "spectacles", "umbrella", "cane", "bonnet",
        "needle", "thread", "scissors", "thimble", "button", "ribbon",
        "kitchen", "pantry", "cellar", "attic", "parlour", "nursery",
        "servant", "housekeeper", "governess", "butler", "maid", "cook",
        # Commerce & finance
        "merchant", "trader", "shopkeeper", "market", "auction", "warehouse",
        "currency", "coin", "banknote", "exchange", "interest", "dividend",
        "bank", "ledger", "invoice", "receipt", "tariff", "customs",
        "insurance", "mortgage", "debt", "credit", "profit", "bankrupt",
        "guild", "apprentice", "journeyman", "master", "wages", "strike",
        "import", "export", "commodity", "wholesale", "retail", "barter",
    ],
    "Geography & Culture": [
        # Places & regions
        "London", "Paris", "Rome", "Constantinople", "Vienna", "Berlin",
        "Edinburgh", "Dublin", "Madrid", "Lisbon", "Amsterdam", "Brussels",
        "Moscow", "Warsaw", "Prague", "Budapest", "Stockholm", "Copenhagen",
        "Calcutta", "Bombay", "Canton", "Peking", "Tokyo", "Cairo",
        "Jerusalem", "Mecca", "Damascus", "Baghdad", "Isfahan", "Delhi",
        "Boston", "Philadelphia", "Washington", "Havana", "Lima", "Mexico",
        # Physical geography
        "ocean", "river", "mountain", "valley", "desert", "forest",
        "island", "peninsula", "cape", "strait", "gulf", "bay",
        "continent", "latitude", "longitude", "equator", "tropics", "arctic",
        "Nile", "Thames", "Danube", "Rhine", "Ganges", "Mississippi",
        "Alps", "Himalayas", "Andes", "Pyrenees", "Caucasus", "Rockies",
        "Atlantic", "Pacific", "Indian", "Mediterranean", "Caribbean", "Baltic",
        # Peoples & nations
        "English", "French", "German", "Spanish", "Italian", "Russian",
        "Chinese", "Japanese", "Indian", "Persian", "Turkish", "Egyptian",
        "American", "Brazilian", "Mexican", "Canadian", "Australian", "African",
        "Christian", "Muslim", "Hindu", "Buddhist", "Jewish", "Confucian",
        "Protestant", "Catholic", "Orthodox", "Presbyterian", "Methodist", "Quaker",
        "Arab", "Malay", "Zulu", "Maori", "Inuit", "Cherokee",
        # Arts & culture
        "painting", "sculpture", "engraving", "fresco", "portrait", "landscape",
        "opera", "symphony", "sonata", "choir", "organ", "violin",
        "theatre", "ballet", "carnival", "festival", "ceremony", "procession",
        "marble", "bronze", "porcelain", "tapestry", "mosaic", "stained",
        "architect", "cathedral", "palace", "castle", "temple", "mosque",
        "Gothic", "Baroque", "Classical", "Romantic", "Renaissance", "Moorish",
    ],
}

# Print counts
for cat, words in WORDS.items():
    # deduplicate
    WORDS[cat] = list(dict.fromkeys(words))
    print(f"  {cat}: {len(WORDS[cat])} words")
total = sum(len(v) for v in WORDS.values())
print(f"  Total: {total} words")

# ---- Collect embeddings ----
print("\n=== Collecting word embeddings ===")
all_words, all_categories = [], []
all_vecs = []

for cat, words in WORDS.items():
    for w in words:
        vec = get_word_embedding(w)
        all_words.append(w)
        all_categories.append(cat)
        all_vecs.append(vec)
    print(f"  {cat}: done ({len(words)} words)")

X = np.stack(all_vecs)
print(f"Embedding matrix: {X.shape}")

# Mean-center
X_centered = X - X.mean(axis=0, keepdims=True)
X_norm = X_centered / (np.linalg.norm(X_centered, axis=1, keepdims=True) + 1e-8)

# ---- UMAP ----
print("\n=== Running UMAP ===")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
umap_2d = reducer.fit_transform(X_norm)

# ---- PCA ----
pca = PCA(n_components=2)
pca_2d = pca.fit_transform(X_norm)
print(f"PCA explained variance: {pca.explained_variance_ratio_[0]:.3f}, {pca.explained_variance_ratio_[1]:.3f}")

# ---- Colors ----
cat_colors = {
    "Sciences": "#e74c3c",
    "Humanities": "#3498db",
    "Daily Life & Technology": "#2ecc71",
    "Geography & Culture": "#f39c12",
}

# ---- Plot 1: UMAP (large, clean) ----
fig, ax = plt.subplots(figsize=(16, 12))
for cat in cat_colors:
    idxs = [i for i, c in enumerate(all_categories) if c == cat]
    ax.scatter(umap_2d[idxs, 0], umap_2d[idxs, 1],
              c=cat_colors[cat], s=20, alpha=0.7, edgecolors='none', label=cat)
ax.legend(fontsize=11, markerscale=2, loc='best')
ax.set_title("UMAP of pre-1900 word embeddings (d34-22btok, final layer)", fontsize=14)
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
fig.savefig(os.path.join(FIGDIR, '11_umap_words.png'), bbox_inches='tight')
print("Saved 11_umap_words.png")
plt.close()

# ---- Plot 2: UMAP with labels on a subset ----
fig, ax = plt.subplots(figsize=(20, 15))
for cat in cat_colors:
    idxs = [i for i, c in enumerate(all_categories) if c == cat]
    ax.scatter(umap_2d[idxs, 0], umap_2d[idxs, 1],
              c=cat_colors[cat], s=25, alpha=0.6, edgecolors='none', label=cat)
# Label every 3rd point to avoid clutter
for i in range(0, len(all_words), 3):
    ax.annotate(all_words[i], (umap_2d[i, 0], umap_2d[i, 1]),
               fontsize=4, alpha=0.6, ha='center', va='bottom',
               xytext=(0, 2), textcoords='offset points')
ax.legend(fontsize=11, markerscale=2, loc='best')
ax.set_title("UMAP of pre-1900 word embeddings (with labels)", fontsize=14)
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
fig.savefig(os.path.join(FIGDIR, '12_umap_words_labeled.png'), bbox_inches='tight')
print("Saved 12_umap_words_labeled.png")
plt.close()

# ---- Plot 3: PCA ----
fig, ax = plt.subplots(figsize=(16, 12))
for cat in cat_colors:
    idxs = [i for i, c in enumerate(all_categories) if c == cat]
    ax.scatter(pca_2d[idxs, 0], pca_2d[idxs, 1],
              c=cat_colors[cat], s=20, alpha=0.7, edgecolors='none', label=cat)
ax.legend(fontsize=11, markerscale=2, loc='best')
ax.set_title("PCA of pre-1900 word embeddings (d34-22btok, final layer)", fontsize=14)
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
fig.savefig(os.path.join(FIGDIR, '13_pca_words.png'), bbox_inches='tight')
print("Saved 13_pca_words.png")
plt.close()

# Cleanup
hook.remove()
print(f"\n=== Done! Figures saved to {FIGDIR} ===")
