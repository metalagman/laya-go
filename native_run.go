package laya

import (
	"context"
	"errors"
	"fmt"
)

type cancelableNativeCall interface {
	Execute() (rawModelOutputs, error)
	Terminate() error
	Close() error
}

type nativeCallResult struct {
	outputs rawModelOutputs
	err     error
}

func executeCancelableNativeCall(
	ctx context.Context,
	call cancelableNativeCall,
) (rawModelOutputs, error) {
	if err := validateContext(ctx); err != nil {
		if call == nil {
			return rawModelOutputs{}, err
		}
		return rawModelOutputs{}, nativeRunCancellationError(err, nil, nil, call.Close())
	}
	if call == nil {
		return rawModelOutputs{}, fmt.Errorf("%w: native call is nil", ErrInvalidState)
	}

	done := make(chan nativeCallResult, 1)
	go func() {
		outputs, err := call.Execute()
		done <- nativeCallResult{outputs: outputs, err: err}
	}()

	select {
	case result := <-done:
		cleanupErr := call.Close()
		if contextErr := ctx.Err(); contextErr != nil {
			return rawModelOutputs{}, nativeRunCancellationError(
				contextErr,
				result.err,
				nil,
				cleanupErr,
			)
		}
		if result.err != nil || cleanupErr != nil {
			return rawModelOutputs{}, errors.Join(result.err, cleanupErr)
		}
		return result.outputs, nil
	case <-ctx.Done():
		terminateErr := call.Terminate()
		result := <-done
		cleanupErr := call.Close()
		return rawModelOutputs{}, nativeRunCancellationError(
			ctx.Err(),
			result.err,
			terminateErr,
			cleanupErr,
		)
	}
}

func nativeRunCancellationError(contextErr error, causes ...error) error {
	joined := []error{contextErr}
	joined = append(joined, causes...)
	return newInferenceStageError(contextErr, "cancel native run", errors.Join(joined...))
}
