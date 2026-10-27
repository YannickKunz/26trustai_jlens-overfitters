# 26trustai_jlens-overfitters

## Getting started -- Local

```
uv sync
```

## Getting started -- Ubelix

Get started with Ubelix: https://github.com/Benji19967/ubelix_hpc/blob/master/README.md

Then, on a submit node run:

```
apptainer build jlens.sif Apptainer.def
```

Request a GPU node: (not enough memory for "Qwen/Qwen3.5-4B", encounters OOM during training. we'll need to try another GPU or reduce the context length of the model)

```
srun --partition=gpu --gres=gpu:rtx4090:1 --cpus-per-task=4 --nodes=1 --mem=16G --time=02:00:00 --pty bash
```

Launch a shell inside the apptainer:

```
apptainer run --nv \
    --bind ./src:/jlens/src \
    jlens.sif
```

Run a python script:

```
Apptainer> python src/main.py
```

## Resources

- [Seminar GitHub Website](https://github.com/lydiaYchen/26TrustAI)
- [J-lens Paper](https://transformer-circuits.pub/2026/workspace/index.html)
- [J-lens++ Paper](https://www.lesswrong.com/posts/nc9dHfcB22JdMzGbr/j-lens-jacobian-filtering-enables-more-faithful-workspace)
- [J-lens GitHub Repo](https://github.com/anthropics/jacobian-lens)
