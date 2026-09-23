package laya

import (
	"context"
	"fmt"
	"io/fs"
	"sync"
)

type runtimeState uint8

const (
	runtimeOpen runtimeState = iota + 1
	runtimeClosing
	runtimeClosed
)

// Runtime coordinates process-local model lifecycles.
//
// A Runtime is safe for simultaneous use by multiple goroutines. Callers own
// every Model returned by OpenModelDir or OpenModelFS and must close all Models
// before closing the Runtime. Runtime.Close refuses to close while a model open
// is in progress or any Model remains live.
type Runtime struct {
	mu sync.Mutex

	state           runtimeState
	engine          runtimeEngine
	models          map[*Model]struct{}
	opening         int
	closeInProgress bool
	closeDone       chan struct{}
}

// NewRuntime validates opts and constructs the process-local native runtime.
// Native builds read the explicit local ONNX Runtime library path documented
// by the native runbook. Builds without the native backend return an error
// matching ErrNativeUnavailable and never substitute a fake implementation.
func NewRuntime(ctx context.Context, opts RuntimeOptions) (*Runtime, error) {
	if err := validateContext(ctx); err != nil {
		return nil, fmt.Errorf("create runtime: %w", err)
	}
	if err := opts.validate(); err != nil {
		return nil, fmt.Errorf("create runtime: %w", err)
	}
	engine, err := newNativeRuntimeEngine(ctx, opts)
	if err != nil {
		return nil, fmt.Errorf("create runtime: %w", err)
	}
	runtime, err := newRuntime(engine)
	if err != nil {
		_ = engine.Close(context.Background())
		return nil, err
	}
	return runtime, nil
}

// OpenModelDir opens a complete model bundle from dir.
//
// The caller retains ownership of dir and must keep its contents immutable for
// the Model lifetime. This method performs no discovery or acquisition.
func (r *Runtime) OpenModelDir(ctx context.Context, dir string, opts ModelOptions) (*Model, error) {
	if err := validateContext(ctx); err != nil {
		return nil, fmt.Errorf("open model directory: %w", err)
	}
	if dir == "" {
		return nil, fmt.Errorf("open model directory: %w: directory is empty", ErrInvalidConfig)
	}
	if err := opts.validate(); err != nil {
		return nil, fmt.Errorf("open model directory: %w", err)
	}

	engine, err := r.beginOpen()
	if err != nil {
		return nil, fmt.Errorf("open model directory: %w", err)
	}
	modelEngine, openErr := engine.OpenModelDir(ctx, dir, opts)
	return r.finishOpen(modelEngine, opts, openErr, "open model directory")
}

// OpenModelFS opens a complete model bundle from fsys.
//
// The caller retains ownership of fsys and must keep it readable and immutable
// until Model.Close completes. Materialization uses the explicit WorkDir in
// opts. A verified digest-addressed publication is retained for later reuse;
// this method performs no discovery, acquisition, or pruning.
func (r *Runtime) OpenModelFS(ctx context.Context, fsys fs.FS, opts FSModelOptions) (*Model, error) {
	if err := validateContext(ctx); err != nil {
		return nil, fmt.Errorf("open model filesystem: %w", err)
	}
	if fsys == nil {
		return nil, fmt.Errorf("open model filesystem: %w: filesystem is nil", ErrInvalidConfig)
	}
	if err := opts.validate(); err != nil {
		return nil, fmt.Errorf("open model filesystem: %w", err)
	}

	engine, err := r.beginOpen()
	if err != nil {
		return nil, fmt.Errorf("open model filesystem: %w", err)
	}
	modelEngine, openErr := engine.OpenModelFS(ctx, fsys, opts)
	return r.finishOpen(modelEngine, opts.ModelOptions, openErr, "open model filesystem")
}

// Close releases Runtime resources after every opened Model has been closed.
//
// Close is safe for concurrent use and is idempotent after success. If Models
// remain open, Close returns an error matching ErrModelsOpen and leaves the
// Runtime open. If ctx ends while cleanup is running, Close returns an error
// matching ctx.Err; a later Close may wait for or retry cleanup.
func (r *Runtime) Close(ctx context.Context) error {
	if ctx == nil {
		return fmt.Errorf("close runtime: %w: context is nil", ErrInvalidRequest)
	}
	if r == nil {
		return fmt.Errorf("close runtime: %w: runtime is nil", ErrInvalidState)
	}

	for {
		r.mu.Lock()
		switch r.state {
		case runtimeClosed:
			r.mu.Unlock()
			return nil
		case runtimeOpen:
			if err := ctx.Err(); err != nil {
				r.mu.Unlock()
				return fmt.Errorf("close runtime: %w", err)
			}
			if r.opening > 0 || len(r.models) > 0 {
				opening := r.opening
				models := len(r.models)
				r.mu.Unlock()
				return fmt.Errorf("close runtime: %w: %d model opens in progress and %d live models", ErrModelsOpen, opening, models)
			}
			r.state = runtimeClosing
		case runtimeClosing:
			if err := ctx.Err(); err != nil {
				r.mu.Unlock()
				return fmt.Errorf("close runtime: %w", err)
			}
		default:
			r.mu.Unlock()
			return fmt.Errorf("close runtime: %w: runtime is not initialized", ErrInvalidState)
		}

		if r.closeInProgress {
			done := r.closeDone
			r.mu.Unlock()
			select {
			case <-done:
				continue
			case <-ctx.Done():
				return fmt.Errorf("close runtime: %w", ctx.Err())
			}
		}

		r.closeInProgress = true
		r.closeDone = make(chan struct{})
		done := r.closeDone
		engine := r.engine
		r.mu.Unlock()

		err := engine.Close(ctx)

		r.mu.Lock()
		r.closeInProgress = false
		if err == nil {
			r.state = runtimeClosed
		}
		close(done)
		r.mu.Unlock()

		if err != nil {
			return fmt.Errorf("close runtime: %w", err)
		}
		return nil
	}
}

func newRuntime(engine runtimeEngine) (*Runtime, error) {
	if engine == nil {
		return nil, fmt.Errorf("create internal runtime: %w: engine is nil", ErrInvalidConfig)
	}
	return &Runtime{
		state:  runtimeOpen,
		engine: engine,
		models: make(map[*Model]struct{}),
	}, nil
}

func (r *Runtime) beginOpen() (runtimeEngine, error) {
	if r == nil {
		return nil, fmt.Errorf("%w: runtime is nil", ErrInvalidState)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.state != runtimeOpen {
		return nil, fmt.Errorf("%w: runtime is not open", ErrInvalidState)
	}
	r.opening++
	return r.engine, nil
}

func (r *Runtime) finishOpen(engine modelEngine, opts ModelOptions, openErr error, operation string) (*Model, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.opening--
	if openErr != nil {
		return nil, fmt.Errorf("%s: %w", operation, openErr)
	}
	if engine == nil {
		return nil, fmt.Errorf("%s: %w: engine returned a nil model", operation, ErrNativeFailure)
	}
	if r.state != runtimeOpen {
		return nil, fmt.Errorf("%s: %w: runtime stopped during open", operation, ErrInvalidState)
	}
	model := newModel(r, engine, opts)
	r.models[model] = struct{}{}
	return model, nil
}

func (r *Runtime) unregister(model *Model) {
	r.mu.Lock()
	delete(r.models, model)
	r.mu.Unlock()
}

type runtimeEngine interface {
	OpenModelDir(context.Context, string, ModelOptions) (modelEngine, error)
	OpenModelFS(context.Context, fs.FS, FSModelOptions) (modelEngine, error)
	Close(context.Context) error
}

func validateContext(ctx context.Context) error {
	if ctx == nil {
		return fmt.Errorf("%w: context is nil", ErrInvalidRequest)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	return nil
}
