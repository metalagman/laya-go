package laya

import (
	"fmt"
	"io/fs"
	"time"
)

// RuntimeOptions configure a Runtime.
//
// The zero value is valid. This foundation release has no runtime-wide knobs;
// the named value keeps construction explicit without introducing option
// functions or mutable package defaults.
type RuntimeOptions struct{}

// TruncationPolicy controls input overflow behavior.
type TruncationPolicy uint8

const (
	// RejectOverflow returns an input-too-long error without changing the input.
	RejectOverflow TruncationPolicy = iota
	// TruncateOverflow enables the pinned Laya-compatible truncation behavior.
	// A successful prediction records the change in TruncationMetadata.
	TruncateOverflow
)

// ModelOptions configure bounded admission and input-budget behavior.
type ModelOptions struct {
	// QueueCapacity is the exact number of callers that may wait behind active
	// inference. Zero rejects callers that cannot start immediately.
	QueueCapacity int
	// MaxQueueWait limits queue waiting in addition to the caller context. Zero
	// adds no timeout; a negative value is invalid.
	MaxQueueWait time.Duration
	// InputTokenLimit optionally lowers the bundle-declared limit. Zero uses the
	// bundle limit; a negative value is invalid.
	InputTokenLimit int
	// Truncation defaults to RejectOverflow.
	Truncation TruncationPolicy
}

// FSModelOptions configure opening a complete bundle from an fs.FS.
//
// Root is a slash-separated fs.ValidPath; an empty Root means ".". WorkDir is
// a required existing caller-owned directory for native-compatible
// materialization. Verified publications below WorkDir are retained for reuse.
type FSModelOptions struct {
	// Root selects the bundle root within the filesystem.
	Root string
	// WorkDir is the caller-owned materialization and retained-publication root.
	WorkDir string
	// ModelOptions apply the common model admission and input policy.
	ModelOptions ModelOptions
}

func (RuntimeOptions) validate() error { return nil }

func (o ModelOptions) validate() error {
	if o.QueueCapacity < 0 {
		return fmt.Errorf("%w: queue capacity cannot be negative", ErrInvalidConfig)
	}
	if o.MaxQueueWait < 0 {
		return fmt.Errorf("%w: maximum queue wait cannot be negative", ErrInvalidConfig)
	}
	if o.InputTokenLimit < 0 {
		return fmt.Errorf("%w: input token limit cannot be negative", ErrInvalidConfig)
	}
	if o.Truncation != RejectOverflow && o.Truncation != TruncateOverflow {
		return fmt.Errorf("%w: unknown truncation policy %d", ErrInvalidConfig, o.Truncation)
	}
	return nil
}

func (o FSModelOptions) validate() error {
	if err := o.ModelOptions.validate(); err != nil {
		return err
	}
	if o.Root != "" && !fs.ValidPath(o.Root) {
		return fmt.Errorf("%w: fs root %q is not a valid path", ErrInvalidConfig, o.Root)
	}
	if o.WorkDir == "" {
		return fmt.Errorf("%w: fs materialization work directory is empty", ErrInvalidConfig)
	}
	return nil
}
