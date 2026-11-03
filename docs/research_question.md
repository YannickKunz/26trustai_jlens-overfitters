# Generalisation of Workspaces Lenses: JLens vs J++Lens

Project guidelines: research questions, hypotheses, methodology, how to interpret the results.

!! It is a draft.

## Background

TODO: Get back the explanation of JLens and J++ lenses from the notes.

We notice that all lenses are fitted on **English WikiText-103**. Behaviour on a change of the corpus fitted on remain untested. This is the gap we want to adress.

## Research Questions

- **RQ1: How can we measure the generalization of a lens?**, Define metrics that permit to see if a lens fitted on a corpus is close to a lens fitted on another.

- **RQ2: Is the J++Lens generalisable across language (English/French) and domain (general text vs mathematics)? How does it compare to the JLens?**

> It may also be interesting to explore how much fitting data is needed before a lens converges.

## Hypotheses

- H1: J++lens is less generalisable than the j-lens

## What do we compare

| Lens | Purpose |
| --- | --- |
| **J-lens** | Baseline (unsupervised) |
| **J++ lens** | Method under test |
| **R-Lens** | Isolates the LRP contribution |

### Corpus

| | General text | Math |
| --- | --- | --- |
| **English** | WikiText-103 | GSM8K question + answer (`openai/gsm8k`, config `main`) |
| **French** | French Wikipedia (`wikimedia/wikipedia`, config `20231101.fr`) | Translated GSM8K |

## Metrics

**Behaviour metric**: recall@10. Following J-Lens++ paper, "an item counts as a hit if its target intermediate variable appears in the top 10 readouts at any of seven equally spaced layers". Tasks: multihop, multilingual, typo, association, poetry. Important: Score all lenses with readout filtering on, otherwise we ll compare the filter, not the lens.

**Distribution metric**: K-L divergence. Between the readout distributions of two lenses. It may be interesting to compare also between a lens and the model's output.

**Parameter metric**: Cosine similarity between flattened matrices of two lens. Frobenius distance, cosine can be dominated by a few large component.

> I've been warn about noise floor, but dont know yet what to do with it.

## Builded matrices

For each method and each layer we build the following matrices.

### Matrix P: performance grid

Rows = corpus the lens was fitted on. Columns = domain it is evaluated on. Entry = recall@10 (mean over eval items).

| fit \ eval | EN text | FR text | EN math | FR math |
| --- | --- | --- | --- | --- |
| EN text | in-dist | transfer | transfer | transfer |
| FR text | transfer | in-dist | transfer | transfer |
| EN math | transfer | transfer | in-dist | transfer |
| FR math | transfer | transfer | transfer | in-dist |

### Matrix S: lens similarity grid

Entry = cosine similarity between lens fitted on corpus f1 and on corpus f2, at layer l. The diagonal is the noise floor(what it is? oh okay, it is two disjoint fit on the same corpus). We have to build also the cross-method version since both have the same shape.

---

Remaining todo:

- [ ] Get back the explanation of JLens and J++ lenses from the notes
- [ ] How to read and interpret the matrices
- [ ] Define noise floor
- [ ] Add the part discussing the convergence of the lens, how many prompt are needed. What curve flatten first and what differ between methods.

Important to check:

- **Same data budget** for every method (positions, layers, sequence length).
- **Same readout filtering** when scoring.
- **No leakage**: the dev set and the evaluation items must be disjoint; fit prompts and evaluation prompts must be disjoint.
- **Several refits** per configuration (different random subsets), since fits vary.
- **Metric bias**: J++ is tuned on latent-variable extraction. If possible add one metric not tied to it (e.g. KL to the model's output).
- **One model is not enough** for a general claim: repeat on a second model if time permits.
