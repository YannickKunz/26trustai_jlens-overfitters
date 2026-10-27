# 26trustai_jlens-overfitters

## Getting started -- Local

```
uv sync
```

## Getting started -- Ubelix

```
apptainer build jlens.sif Apptainer.def
```

Not enough memory for "Qwen/Qwen3.5-4B", encounters OOM during training. 
We'll need to try another GPU or reduce the context length of the model.

```
srun --partition=gpu --gres=gpu:rtx4090:1 --cpus-per-task=4 --nodes=1 --mem=16G --time=02:00:00 --pty bash
```

```
apptainer run --nv \
    --bind ./src:/jlens/src \
    jlens.sif
```

```
Apptainer> python src/main.py
```

## Resources

- [Seminar GitHub Website](https://github.com/lydiaYchen/26TrustAI)
- [J-lens Paper](https://transformer-circuits.pub/2026/workspace/index.html)
- [J-lens++ Paper](https://www.lesswrong.com/posts/nc9dHfcB22JdMzGbr/j-lens-jacobian-filtering-enables-more-faithful-workspace)
- [J-lens GitHub Repo](https://github.com/anthropics/jacobian-lens)
