# Release checks

The packaged code contains no model weights, downloaded datasets, server
paths, GPU UUIDs, usernames, credentials, or experiment job logs.

Before claiming exact reproduction of the submitted paper numbers, run these
checks on the experiment server or against exported artifacts:

1. Compare SHA-256 of the train/dev/confirmation JSONL files and the teacher
   and student adapter weights with the frozen experiment manifests.
2. Run one seed of `KD-2view` and `ERD` through the evaluation command and
   compare per-example cosine, RelL2, paired success, and candidate success
   with archived outputs. Compare *values*, not just aggregate means.
3. Verify the SeqKD teacher-generation cache hash and confirm all methods
   receive the same 512 confirmation IDs in the same order.
4. Confirm LoRA initialization, installed package versions, and tokenizer
   compatibility. Newer libraries or GPUs may change floating-point results.
5. Review the MIT copyright line and upstream HotpotQA/Meta model terms.

The clean training CLI is a faithful implementation of the core formulas and
paper hyperparameters, not a byte-for-byte copy of the historical queue
runner. The separate teacher trainer is a reference implementation because
the original standalone teacher SFT script was not available locally.
