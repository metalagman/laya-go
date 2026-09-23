// Package directory demonstrates inference from a complete caller-owned local
// bundle directory, including a complete externally cached HF snapshot.
package directory

import (
	"context"
	"fmt"
	"io"

	"github.com/metalagman/laya-go"
	"github.com/metalagman/laya-go/examples/internal/exampleapp"
)

// Run opens one local bundle, evaluates all primitives, and closes Model before
// Runtime. It never discovers or downloads model files.
func Run(ctx context.Context, bundleDir string, state laya.State, output io.Writer) error {
	runtime, err := laya.NewRuntime(ctx, laya.RuntimeOptions{})
	if err != nil {
		return fmt.Errorf("create runtime: %w", err)
	}
	model, err := runtime.OpenModelDir(ctx, bundleDir, laya.ModelOptions{QueueCapacity: 4, Truncation: laya.TruncateOverflow})
	if err != nil {
		_ = runtime.Close(context.Background())
		return fmt.Errorf("open local bundle: %w", err)
	}
	runErr := exampleapp.Evaluate(ctx, model, state, output)
	modelErr := model.Close(context.Background())
	runtimeErr := runtime.Close(context.Background())
	if runErr != nil {
		return runErr
	}
	if modelErr != nil {
		return fmt.Errorf("close model: %w", modelErr)
	}
	if runtimeErr != nil {
		return fmt.Errorf("close runtime: %w", runtimeErr)
	}
	return nil
}
