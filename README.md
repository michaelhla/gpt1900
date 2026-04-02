# GPT-1900

<p align="center">
  <img src="figures/gpt1900_banner.png" width="700" alt="GPT-1900">
</p>

<p align="center">
  <em>An experiment to see if an LLM trained from scratch on text prior to 1900 can come up with quantum mechanics and relativity.</em>
</p>

<p align="center">
  <a href="https://huggingface.co/collections/mhla/gpt-1900"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-mhla/gpt--1900-yellow" alt="Hugging Face"></a>
  <a href="https://michaelhla.com/blog/machina-mirabilis.html"><img src="https://img.shields.io/badge/Blog-Machina%20Mirabilis-red" alt="Blog Post"></a>
  <a href="https://x.com/hla_michael"><img src="https://img.shields.io/badge/Twitter-@hla__michael-1DA1F2?logo=twitter&style=social" alt="Twitter"></a>
</p>

## Key Resources

- **Eval prompts**: [`EVAL.json`](EVAL.json) (with classical assumptions) · [`EVAL_no_assumptions.json`](EVAL_no_assumptions.json) (without)
- **Generations**:
  - [Best generations (v11)](results/physics_eval_v11/best_generations.md) · [all generations](results/physics_eval_v11/generations.json) · [judged results](results/physics_eval_v11/results_judged.json)
  - [No-assumptions variant](results/physics_eval_v11_no_assumptions/generations.json) · [rephrased prompts](results/physics_eval_v11_rephrase/generations.json) · [replication](results/physics_eval_v11_replication/generations.json)
  - [Base model, no SFT RL (v6)](results/physics_eval_v6/generations.json)
  - [Gallery of funny failures](results/gallery_of_funny_failures.md) — curated highlights across all runs

## Chat with GPT-1900

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/mhla/gpt1900.git
cd gpt1900
uv sync --extra gpu    # or --extra cpu for CPU / Apple Silicon
source .venv/bin/activate

# Download and chat (instruction-tuned model)
bash runs/chat.sh

# Chat with the RL model
bash runs/chat.sh -r mhla/gpt1900-d34-contradiction-rl-v11
```

## License

MIT
