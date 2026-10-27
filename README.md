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

Request a GPU node: (Using "Qwen/Qwen3.5-4B", can fit a J-Lens using `dim_batch=8`, otherwise encountering OOM error. Would be ideal to try running on a more powerful GPU. With this setup, training takes ~50s per prompt.)

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
- [Neuronpedia J-Lens](https://www.neuronpedia.org/jlens)
