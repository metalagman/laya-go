// Package laya defines explicit, process-local contracts for Laya decision
// models.
//
// A caller constructs a Runtime, opens each Model from a caller-owned source,
// closes every Model, and then closes the Runtime. Runtime and Model are safe
// for concurrent use. Blocking operations accept a context; contexts are not
// retained after a call.
//
// Predict evaluates one immutable State against an ordered, heterogeneous set
// of Choice, Score, and Noul questions. Its Prediction preserves result order
// and reports Usage once for the complete batch. Text State preserves its input
// exactly. JSON State validates, copies, and compacts its input.
//
// Model acquisition is outside the package. A local directory or fs.FS must
// contain a complete Laya Go Bundle. It may be an application asset, embed.FS,
// or an externally acquired Hugging Face snapshot in the narrowly documented
// snapshot/blob layout. A raw Safetensors checkpoint is build-time exporter
// input, not a runtime bundle. Callers keep source content immutable through
// Model.Close. The package does not authenticate, download, select revisions,
// or manage external caches.
//
// An ordinary build has no native dependencies, so NewRuntime returns an error
// matching ErrNativeUnavailable. An explicit laya_native cgo build can use the
// reviewed native-runtime candidate with caller-supplied pinned libraries and
// a verified local bundle. OpenModelFS materializes already-present bytes below
// its explicit caller-owned WorkDir; digest-addressed publications are retained
// for verified reuse. Current source and native compatibility evidence is
// limited to linux/amd64. ADK integration remains unimplemented. The package
// never substitutes fake inference behavior.
package laya
