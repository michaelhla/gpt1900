# GPT-1900

<p align="center">
  <img src="figures/machina_mirabilis.png" width="700" alt="Machina Mirabilis">
</p>

<p align="center">
  <em>An experiment to see if an LLM trained from scratch on text prior to 1900 can come up with quantum mechanics and relativity.</em>
</p>

<p align="center">
  <a href="https://huggingface.co/collections/mhla/gpt-1900"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-mhla/gpt--1900-yellow" alt="Hugging Face"></a>
  <a href="https://michaelhla.com/blog/machina-mirabilis.html"><img src="https://img.shields.io/badge/Blog-Machina%20Mirabilis-red" alt="Blog Post"></a>
  <a href="https://x.com/hla_michael"><img src="https://img.shields.io/badge/Twitter-@hla__michael-1DA1F2?logo=twitter&style=social" alt="Twitter"></a>
</p>

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
