package laya

import (
	"errors"
	"fmt"
	"time"

	internalbundle "github.com/metalagman/laya-go/internal/bundle"
	internalsource "github.com/metalagman/laya-go/internal/source"
)

var (
	// ErrInvalidConfig indicates caller-supplied configuration is invalid.
	ErrInvalidConfig = errors.New("laya: invalid configuration")
	// ErrInvalidState indicates a State is malformed or was not constructed.
	ErrInvalidState = errors.New("laya: invalid state")
	// ErrModelsOpen indicates a Runtime cannot close while it owns live Models.
	ErrModelsOpen = errors.New("laya: models are open")
	// ErrModelClosed indicates an operation targeted a closed Model.
	ErrModelClosed = errors.New("laya: model is closed")
	// ErrInvalidRequest indicates a question or prediction request is invalid.
	ErrInvalidRequest = errors.New("laya: invalid request")
	// ErrInputTooLong indicates input exceeds the effective token limit.
	ErrInputTooLong = errors.New("laya: input is too long")
	// ErrQueueFull indicates bounded admission has no available queue slot.
	ErrQueueFull = errors.New("laya: queue is full")
	// ErrQueueTimeout indicates a caller exceeded its configured queue wait.
	ErrQueueTimeout = errors.New("laya: queue wait timed out")
	// ErrInvalidBundle indicates a supplied model bundle is invalid.
	ErrInvalidBundle = internalbundle.ErrInvalid
	// ErrIntegrity indicates a model bundle artifact does not match its identity.
	ErrIntegrity = internalbundle.ErrIntegrity
	// ErrUnsupportedBundle indicates valid bundle metadata is not supported.
	ErrUnsupportedBundle = internalbundle.ErrUnsupported
	// ErrUnsupportedSource indicates a supplied bundle source is incompatible.
	ErrUnsupportedSource = internalsource.ErrUnsupported
	// ErrMaterialization indicates local filesystem staging or publication failed.
	ErrMaterialization = internalsource.ErrMaterialization
	// ErrNativeUnavailable indicates the native inference engine is unavailable.
	ErrNativeUnavailable = errors.New("laya: native inference is unavailable")
	// ErrNativeFailure indicates the native inference engine failed.
	ErrNativeFailure = errors.New("laya: native inference failure")
	// ErrInvalidOutput indicates model output violates the public result contract.
	ErrInvalidOutput = errors.New("laya: invalid model output")
)

// BundleError describes a bundle path, field, or compatibility failure while
// preserving its caller-actionable category and any underlying cause. Its
// metadata is redacted and never contains model, tokenizer, State, or tensor
// content.
type BundleError = internalbundle.Error

// InputTooLongError reports the observed and allowed token counts.
//
// It matches ErrInputTooLong with errors.Is. If Err is non-nil, Unwrap exposes
// that underlying cause as well.
type InputTooLongError struct {
	// InputTokens is the observed input token count.
	InputTokens int
	// Limit is the effective maximum input token count.
	Limit int
	// Err is an optional underlying cause.
	Err error
}

// Error returns a human-readable input-budget failure.
func (e *InputTooLongError) Error() string {
	if e == nil {
		return ErrInputTooLong.Error()
	}
	if e.Err != nil {
		return fmt.Sprintf("%s: input tokens %d exceed limit %d: %v", ErrInputTooLong, e.InputTokens, e.Limit, e.Err)
	}
	return fmt.Sprintf("%s: input tokens %d exceed limit %d", ErrInputTooLong, e.InputTokens, e.Limit)
}

// Is reports whether target is ErrInputTooLong.
func (e *InputTooLongError) Is(target error) bool {
	return e != nil && target == ErrInputTooLong
}

// Unwrap returns the underlying cause, if any.
func (e *InputTooLongError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

// QueueErrorKind identifies a bounded-admission failure.
type QueueErrorKind uint8

const (
	// QueueFull means no configured queue slot was available.
	QueueFull QueueErrorKind = iota + 1
	// QueueWaitTimeout means the configured maximum queue wait elapsed.
	QueueWaitTimeout
)

// QueueError reports a bounded-admission failure and preserves its cause.
type QueueError struct {
	// Kind identifies whether admission failed immediately or after waiting.
	Kind QueueErrorKind
	// Capacity is the configured number of waiting callers.
	Capacity int
	// Wait is the configured maximum wait for timeout errors.
	Wait time.Duration
	// Err is an optional underlying cause.
	Err error
}

// Error returns a human-readable queue failure.
func (e *QueueError) Error() string {
	if e == nil {
		return "laya: queue failure"
	}
	category := e.category()
	if category == nil {
		return fmt.Sprintf("laya: invalid queue failure kind %d", e.Kind)
	}
	if e.Err != nil {
		return fmt.Sprintf("%s: capacity %d after %s: %v", category, e.Capacity, e.Wait, e.Err)
	}
	return fmt.Sprintf("%s: capacity %d after %s", category, e.Capacity, e.Wait)
}

// Is reports whether target is the sentinel for the queue failure kind.
func (e *QueueError) Is(target error) bool {
	category := e.category()
	return category != nil && target == category
}

// Unwrap returns the underlying cause, if any.
func (e *QueueError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

func (e *QueueError) category() error {
	if e == nil {
		return nil
	}
	switch e.Kind {
	case QueueFull:
		return ErrQueueFull
	case QueueWaitTimeout:
		return ErrQueueTimeout
	default:
		return nil
	}
}
