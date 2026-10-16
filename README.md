# Evidence-Response Distillation (ERD)

Research code for the ERD paper. ERD aligns a student's **change in centered
candidate-answer scores** under an original/edited evidence pair with the
teacher's change. The released implementation includes the HotpotQA mechanical
pair builder, the response and two-view KD losses, matched training controls,
the SeqKD baseline, and confirmation-set evaluation.

The code uses **ERD** as the public method name. Historical result files may
use `A-CET`; the names refer to the same objective in this paper.

## What is included

| Command | Purpose |
| --- | --- |
| `python -m erd.prepare_data` | Construct train/dev/test answer-replacement pairs and 32-answer candidate sets. |
| `python -m erd.train_teacher` | Reference two-view SFT to prepare an adapted teacher. |
| `python -m erd.train_student` | Train ERD, SFT, KD, SeqKD, shuffled, or loss ablations. |
| `python -m erd.cache_seqkd` | Freeze teacher greedy generations for SeqKD. |
| `python -m erd.evaluate` | Score response fidelity, candidate ranking, and free generation. |
| `python -m erd.summarize` | Aggregate seeds and compute within-seed paired-example intervals. |

The repository does not contain HotpotQA data, Meta model weights, LoRA
checkpoints, or generated predictions. Obtain those artifacts under their own
licenses and access conditions. The original 512-pair confirmation manifest is
also not bundled in this code-only release; see **Reproduction scope** below.

## Installation

Use Python 3.11 and a PyTorch build suitable for your CUDA installation.
The historical server used Python 3.11.16, PyTorch 2.13.0+cu130,
Transformers 4.57.6, PEFT 0.14.0, and NumPy 2.4.6. Install your CUDA-compatible
PyTorch first, then:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

Set the following shell variables to local paths (examples only):

```bash
export HOTPOTQA=/path/to/HotpotQA
export TEACHER_BASE=/path/to/Llama-3.1-8B-Instruct
export STUDENT_BASE=/path/to/llama3.2-1b-instruct
export WORKDIR=/path/to/erd-work
```

The HotpotQA directory must contain `distractor/train-*.parquet` and
`fullwiki/validation-00000-of-00001.parquet`. Model directories must be in
Hugging Face format and available locally. The paper's teacher and student are
both Llama models with compatible token-ID mappings; the code checks this
before training/evaluation.

## Data construction

```bash
python -m erd.prepare_data --hotpotqa "$HOTPOTQA" --output "$WORKDIR/data"
```

This reproduces the archived *mechanical* construction rule: replace answer
strings in evidence, choose a foil from other answers, and build a shared
32-candidate answer list. It is not a semantic-validity filter. The builder
writes a manifest with source counts and SHA-256 hashes; keep it with results.
The generated `test.jsonl` is from HotpotQA fullwiki validation. Do not use it
to select hyperparameters.

For the exact reported confirmation experiment, evaluate the separately
frozen 512-pair JSONL rather than rebuilding a different held-out set.

## Teacher preparation

If an adapted teacher adapter is available, use it directly. Otherwise the
following reference implementation trains a two-view LoRA teacher:

```bash
python -m erd.train_teacher \
  --train "$WORKDIR/data/train.jsonl" \
  --teacher-base "$TEACHER_BASE" \
  --output "$WORKDIR/teacher" \
  --device cuda:0
export TEACHER_ADAPTER="$WORKDIR/teacher/adapter"
```

The historical paper checkpoint was selected at 1.5 epochs/384 steps. The
standalone original teacher-training runner was not present in the writing
handoff, so this reference command is **not claimed to reproduce that adapter
bit for bit**. Exact numerical reproduction requires the original frozen
teacher adapter.

## Student training

The paper configuration uses Llama-3.2-1B-Instruct, rank-8 LoRA on
`q_proj,v_proj,o_proj`, 1,000 updates, two pairs per update, AdamW at
`1e-4`, 50 warmup updates, `alpha=1`, `beta=0.5`, `gamma=0.05`, and seeds
71/73/79. Each run saves an adapter, development curve, and input hashes.

```bash
for seed in 71 73 79; do
  for method in KD-2view ERD ERD-shuffled ERD-direction ERD-magnitude; do
    python -m erd.train_student \
      --train "$WORKDIR/data/train.jsonl" \
      --dev "$WORKDIR/data/dev.jsonl" \
      --teacher-base "$TEACHER_BASE" \
      --teacher-adapter "$TEACHER_ADAPTER" \
      --student-base "$STUDENT_BASE" \
      --method "$method" --seed "$seed" \
      --output "$WORKDIR/runs/$method/seed$seed" \
      --device cuda:0
  done
done
```

Optional methods: `SFT-original`, `SFT-2view-upper`, `KD-1view`,
`SeqKD-2view`. The two-view SFT condition is an upper-bound diagnostic, not a
distillation baseline. For SeqKD,
first freeze the teacher generations:

```bash
python -m erd.cache_seqkd \
  --train "$WORKDIR/data/train.jsonl" \
  --teacher-base "$TEACHER_BASE" \
  --teacher-adapter "$TEACHER_ADAPTER" \
  --student-base "$STUDENT_BASE" \
  --output "$WORKDIR/seqkd_cache.json" --device cuda:0
```

Then run `train_student --method SeqKD-2view --seqkd-cache
"$WORKDIR/seqkd_cache.json"` with the other arguments above. Match the
teacher cache across all SeqKD seeds.

The `ERD-shuffled` control deranges teacher response targets within each
two-pair update. ERD direction and magnitude ablations respectively set
`gamma=0` and `beta=0` while keeping all other settings fixed.

## Evaluation and aggregation

Run each frozen adapter on the same held-out JSONL. The example below assumes
the published confirmation pairs have been placed at
`$WORKDIR/data/confirm512.jsonl`:

```bash
for seed in 71 73 79; do
  for method in KD-2view ERD ERD-shuffled ERD-direction ERD-magnitude SeqKD-2view; do
    python -m erd.evaluate \
      --pairs "$WORKDIR/data/confirm512.jsonl" \
      --teacher-base "$TEACHER_BASE" \
      --teacher-adapter "$TEACHER_ADAPTER" \
      --student-base "$STUDENT_BASE" \
      --student-adapter "$WORKDIR/runs/$method/seed$seed/adapter" \
      --output "$WORKDIR/runs/$method/seed$seed/eval" \
      --device cuda:0
  done
done

python -m erd.summarize --results "$WORKDIR/runs" --output "$WORKDIR/tables"
```

`paired_success` is the fraction of pairs for which **both views have token
F1 strictly greater than 0.5**. The historical output key `paired_f1` is an
alias for that binary indicator; it is not a continuous F1 score. Candidate
accuracy is measured separately from greedy free generation. Response cosine
uses `eps=1e-8` to clamp each vector norm, so zero responses remain in the
average. Relative L2 divides by `teacher_norm + 1e-8`.

The paired bootstrap resamples examples within each trained seed. Its intervals
do not establish significance across training seeds.

## Reproduction scope and provenance

This is a portable, code-only extraction of the archived E52/E56/E57/E58
experiment path. We preserved the answer scoring, response, optimization, and
evaluation definitions. The original scripts mixed those definitions with
absolute server paths, GPU UUIDs, job queues, and one-off decision gates;
those orchestration details are not part of this release.

The release has three limits to keep visible:

1. The original teacher adapter and complete E52 train/dev pair files were
   not in the local writing handoff. Running the builder/reference teacher
   trainer creates new artifacts that may not be byte-identical to the paper's.
2. The independent 512-pair confirmation set is not included. Exact numerical
   verification needs its frozen JSONL and original model checkpoints.
3. The earlier synthetic paired-association diagnostic used a different
   legacy pipeline and is not included in this HotpotQA-focused package.
4. GKD's matched run ended in out-of-memory errors and is not presented as a
   reproduced baseline here.

No generated results in this repository should be represented as the paper's
reported scores until checkpoint and dataset hashes are matched. See
`RELEASE_NOTES.md` for the pre-publication parity checks.

## License

Code is distributed under the MIT License. Dataset and model licenses are
separate. Confirm the author/copyright attribution in `LICENSE` before public
publication.
