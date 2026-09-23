// Package embedded demonstrates inference from a caller-owned fs.FS, including
// embed.FS, materialized into explicit caller-selected work storage.
package embedded

import (
	"context"
	"fmt"
	"io"
	"io/fs"

	"github.com/metalagman/laya-go"
	"github.com/metalagman/laya-go/examples/internal/exampleapp"
)

// Run stages one complete filesystem bundle, evaluates all primitives, and
// closes Model before Runtime. WorkDir must already exist.
func Run(ctx context.Context, bundle fs.FS, workDir string, state laya.State, output io.Writer) error {
	runtime, err := laya.NewRuntime(ctx, laya.RuntimeOptions{})
	if err != nil {
		return fmt.Errorf("create runtime: %w", err)
	}
	model, err := runtime.OpenModelFS(ctx, bundle, laya.FSModelOptions{WorkDir: workDir, ModelOptions: laya.ModelOptions{QueueCapacity: 4, Truncation: laya.TruncateOverflow}})
	if err != nil {
		_ = runtime.Close(context.Background())
		return fmt.Errorf("open filesystem bundle: %w", err)
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
