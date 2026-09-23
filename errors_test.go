package laya

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestBundleErrorCategories(t *testing.T) {
	cause := errors.New("underlying")
	err := &BundleError{
		Path:     "manifest.json",
		Field:    "model.precision",
		Expected: "fp32",
		Observed: "fp16",
		Kind:     ErrUnsupportedBundle,
		Cause:    cause,
	}
	if !errors.Is(err, ErrUnsupportedBundle) {
		t.Errorf("errors.Is(BundleError, ErrUnsupportedBundle) = false")
	}
	if !errors.Is(err, cause) {
		t.Errorf("errors.Is(BundleError, cause) = false")
	}
	var got *BundleError
	if !errors.As(err, &got) || got.Field != "model.precision" {
		t.Errorf("errors.As(BundleError) = %#v, want field model.precision", got)
	}
	if errors.Is(err, ErrInvalidBundle) || errors.Is(err, ErrIntegrity) {
		t.Errorf("BundleError unexpectedly matches unrelated category")
	}
}

func TestInputTooLongErrorSupportsIsAsAndUnwrap(t *testing.T) {
	want := &InputTooLongError{
		InputTokens: 1024,
		Limit:       512,
		Err:         context.Canceled,
	}
	err := error(want)

	if !errors.Is(err, ErrInputTooLong) {
		t.Errorf("errors.Is(%v, ErrInputTooLong) = false, want true", err)
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("errors.Is(%v, context.Canceled) = false, want true", err)
	}
	var got *InputTooLongError
	if !errors.As(err, &got) || got.InputTokens != 1024 || got.Limit != 512 {
		t.Errorf("errors.As(%v, *InputTooLongError) = %+v, want token counts 1024 and 512", err, got)
	}
	if got, want := err.Error(), "laya: input is too long: input tokens 1024 exceed limit 512: context canceled"; got != want {
		t.Errorf("err.Error() = %q, want %q", got, want)
	}

	withoutCause := &InputTooLongError{InputTokens: 2, Limit: 1}
	if got, want := withoutCause.Error(), "laya: input is too long: input tokens 2 exceed limit 1"; got != want {
		t.Errorf("withoutCause.Error() = %q, want %q", got, want)
	}
	var nilInput *InputTooLongError
	if errors.Is(nilInput, ErrInputTooLong) {
		t.Errorf("errors.Is((*InputTooLongError)(nil), ErrInputTooLong) = true, want false")
	}
	if got, want := nilInput.Error(), ErrInputTooLong.Error(); got != want {
		t.Errorf("((*InputTooLongError)(nil)).Error() = %q, want %q", got, want)
	}
	if got := nilInput.Unwrap(); got != nil {
		t.Errorf("((*InputTooLongError)(nil)).Unwrap() = %v, want nil", got)
	}
}

func TestQueueErrorSupportsIsAsAndUnwrap(t *testing.T) {
	tests := []struct {
		name string
		err  *QueueError
		want error
	}{
		{
			name: "full",
			err:  &QueueError{Kind: QueueFull, Capacity: 2},
			want: ErrQueueFull,
		},
		{
			name: "timeout",
			err:  &QueueError{Kind: QueueWaitTimeout, Capacity: 2, Wait: time.Second, Err: context.DeadlineExceeded},
			want: ErrQueueTimeout,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if !errors.Is(test.err, test.want) {
				t.Errorf("errors.Is(%v, %v) = false, want true", test.err, test.want)
			}
			if test.err.Err != nil && !errors.Is(test.err, test.err.Err) {
				t.Errorf("errors.Is(%v, %v) = false, want true", test.err, test.err.Err)
			}
			var got *QueueError
			if !errors.As(test.err, &got) || got.Kind != test.err.Kind {
				t.Errorf("errors.As(%v, *QueueError) = %+v, want kind %v", test.err, got, test.err.Kind)
			}
			if got := test.err.Error(); got == "" {
				t.Errorf("test.err.Error() is empty, want diagnostic text")
			}
		})
	}

	invalid := &QueueError{}
	if errors.Is(invalid, ErrQueueFull) || errors.Is(invalid, ErrQueueTimeout) {
		t.Errorf("errors.Is(%v, queue sentinel) = true, want false for invalid kind", invalid)
	}
	if got, want := invalid.Error(), "laya: invalid queue failure kind 0"; got != want {
		t.Errorf("invalid.Error() = %q, want %q", got, want)
	}
	var nilQueue *QueueError
	if got, want := nilQueue.Error(), "laya: queue failure"; got != want {
		t.Errorf("((*QueueError)(nil)).Error() = %q, want %q", got, want)
	}
	if got := nilQueue.Unwrap(); got != nil {
		t.Errorf("((*QueueError)(nil)).Unwrap() = %v, want nil", got)
	}
}
