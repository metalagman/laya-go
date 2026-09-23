# Laya upstream parity v1

Preprocessing and output behavior are pinned to `NandhaKishorM/laya` version
0.3.5 at commit `573e5b62696ba441230cd6be71d593331b5d23af` and the official multilingual
checkpoint revision `052592a15d198d9ad47da779604259b10b47b7aa`.

## State and questions

Text State is preserved byte-for-byte. JSON State is the exact compact UTF-8
representation produced by the public Go constructor and is passed to the
reference as a string, never reparsed into a Python dict or list.

Choice criteria and Score levels preserve caller order. Choice renders the ID
and description; Score renders the zero-based level and description. Noul uses
the fixed false/true order and pinned defaults when descriptions are empty.
Occurrences of the tokenizer mask token in model-facing text are replaced by
one ASCII space before tokenization.

## Sequence and batch

Each question uses this sequence:

```text
[CLS]
tokens("<qtype> question: <instructions>")
[SEP]
for each option: [MASK] + tokens(" " + rendered option)[:48]
[SEP]
state tokens fitting max_len
[SEP]
```

`max_len` is 1024, `head_max_len` is 256, the head reserve is 16, and the
option-body cap is 48. Compatibility truncation keeps the State prefix. Public
inference rejects overflow by default and must never silently discard an option
marker.

Questions remain in input order. Sequence and marker axes pad independently;
input IDs use the tokenizer pad ID, attention and marker masks use zero/false,
marker positions use zero for padded entries, and qtype is choice=0, score=1,
noul=2.

## Outputs

The complete graph returns `logits` and `act_logits`. Output names, shapes,
dtypes, and finiteness are validated before formatting. Calibration selects an
option-count bucket before the qtype temperature, clamps finite temperatures to
[0.5, 5.0], and uses 1.0 for an invalid value. Softmax is numerically stable.

Choice and Score confidence use normalized entropy. Score is the expected
zero-based level. Noul reports P(true) at index 1. Action probability is a
separate softmax result at the manifest-declared action index and is never
treated as answer confidence. Public scalar values round to four decimals.

FP32 comparison happens before rounding with initial `atol=1e-4` and
`rtol=1e-4`. Tensor shapes, token IDs, masks, ordering, selected answers, and
rounded public values match exactly. A failed tolerance requires investigation;
it does not authorize silent widening or artifact substitution.

## Frozen synthetic corpus

`internal/parity/testdata/corpus-v1.json` is the small committed parity oracle.
It contains synthetic text only and binds the exact official checkpoint, SDK,
profile, exporter source, dependency lock, manifest, and all nine bundle file
identities. It stores exact rendered options, token IDs, all five padded input
tensors, reference and ONNX `logits`/`act_logits`, calibration, pre-rounding
values, typed results, and shared usage.

The reviewed inventory covers English, Russian, mixed text, combining/RTL/CJK/
emoji Unicode, mask replacement, Choice/Score/Noul, 2/6/11 option buckets,
JSON primitives, ordered batches, long options, exact fit, explicit right
truncation, default overflow rejection, and impossible marker layouts. The Go
loader rejects unknown fields, inventory drift, identity/tolerance changes,
non-finite or malformed tensors, reference/ONNX tolerance failures, and any
change to the committed corpus digest. `task reference:verify` is hermetic and
requires no Python, model, native runtime, or network.

Regeneration is deliberately opt-in and local-only:

```sh
LAYA_SOURCE_DIR=/absolute/official-snapshot \
LAYA_SDK_DIR=/absolute/pinned-sdk \
LAYA_BUNDLE_DIR=/absolute/complete-bundle \
LAYA_FIXTURE_OUTPUT=/absolute/corpus-v1.json \
task reference:generate
```

The command verifies all supplied identities before model construction, runs
the repository architecture and accepted ONNX graph over three batches, and
writes the selected output atomically. It never resolves a repository ID,
downloads a model, mutates an external cache, or publishes an artifact.
