# Local-only tokenizer binding

This package is a reduced derivative of
`github.com/daulet/tokenizers/tokenizer.go` and `tokenizers.h` at tag
`v1.27.0`, commit `f678a7768d5479d9d5a5161c4fc45c8a5ba46146`.

Retained capabilities are limited to parsing an already-local tokenizer,
encoding text, reporting the linked version, and cleanup. Hugging Face model
resolution, HTTP, authentication, downloads, cache management, decoding,
Tiktoken, truncation configuration, and unused output attributes were removed.

The upstream MIT license is preserved in `LICENSE.upstream`.
