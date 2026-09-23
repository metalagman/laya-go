package laya

import (
	"context"
	"errors"
	"testing"
	"testing/fstest"
)

func TestNewRuntimeReportsNativeUnavailable(t *testing.T) {
	t.Setenv("LAYA_ONNXRUNTIME_LIBRARY", "")
	runtime, err := NewRuntime(t.Context(), RuntimeOptions{})
	if runtime != nil {
		t.Errorf("NewRuntime() runtime = %v, want nil", runtime)
	}
	if !errors.Is(err, ErrNativeUnavailable) {
		t.Errorf("NewRuntime() error = %v, want errors.Is(_, ErrNativeUnavailable)", err)
	}

	canceled, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := NewRuntime(canceled, RuntimeOptions{}); !errors.Is(err, context.Canceled) {
		t.Errorf("NewRuntime(canceled) error = %v, want errors.Is(_, context.Canceled)", err)
	}
	var nilContext context.Context
	if _, err := NewRuntime(nilContext, RuntimeOptions{}); !errors.Is(err, ErrInvalidRequest) {
		t.Errorf("NewRuntime(nil) error = %v, want errors.Is(_, ErrInvalidRequest)", err)
	}
}

func TestRuntimeOwnsModelLifecycle(t *testing.T) {
	modelEngine := &fakeModelEngine{predict: func(_ context.Context, _ State, questions []Question) (Prediction, error) {
		return fakePrediction(questions)
	}}
	engine := &fakeRuntimeEngine{dirModel: modelEngine}
	runtime, err := newRuntime(engine)
	if err != nil {
		t.Fatalf("newRuntime() returned unexpected error: %v", err)
	}
	model, err := runtime.OpenModelDir(t.Context(), "/models/laya", ModelOptions{})
	if err != nil {
		t.Fatalf("runtime.OpenModelDir() returned unexpected error: %v", err)
	}

	if err := runtime.Close(t.Context()); !errors.Is(err, ErrModelsOpen) {
		t.Errorf("runtime.Close() with a live model = %v, want errors.Is(_, ErrModelsOpen)", err)
	}
	if err := model.Close(t.Context()); err != nil {
		t.Fatalf("model.Close() returned unexpected error: %v", err)
	}
	if err := model.Close(t.Context()); err != nil {
		t.Errorf("second model.Close() = %v, want nil", err)
	}
	if err := runtime.Close(t.Context()); err != nil {
		t.Fatalf("runtime.Close() returned unexpected error: %v", err)
	}
	if err := runtime.Close(t.Context()); err != nil {
		t.Errorf("second runtime.Close() = %v, want nil", err)
	}
	canceled, cancel := context.WithCancel(t.Context())
	cancel()
	if err := runtime.Close(canceled); err != nil {
		t.Errorf("closed runtime.Close(canceled) = %v, want nil", err)
	}
	if _, err := runtime.OpenModelDir(t.Context(), "/models/other", ModelOptions{}); !errors.Is(err, ErrInvalidState) {
		t.Errorf("runtime.OpenModelDir() after close error = %v, want errors.Is(_, ErrInvalidState)", err)
	}

	dirCalls, fsCalls, closeCalls := engine.counts()
	if dirCalls != 1 || fsCalls != 0 || closeCalls != 1 {
		t.Errorf("runtime engine calls = (dir %d, fs %d, close %d), want (1, 0, 1)", dirCalls, fsCalls, closeCalls)
	}
	_, modelCloseCalls := modelEngine.counts()
	if modelCloseCalls != 1 {
		t.Errorf("model engine close calls = %d, want 1", modelCloseCalls)
	}
}

func TestRuntimeZeroValueFailsClosed(t *testing.T) {
	var runtime Runtime
	if err := runtime.Close(t.Context()); !errors.Is(err, ErrInvalidState) {
		t.Errorf("(Runtime{}).Close() error = %v, want errors.Is(_, ErrInvalidState)", err)
	}
	if _, err := runtime.OpenModelDir(t.Context(), "/model", ModelOptions{}); !errors.Is(err, ErrInvalidState) {
		t.Errorf("(Runtime{}).OpenModelDir() error = %v, want errors.Is(_, ErrInvalidState)", err)
	}
}

func TestRuntimeOpensCallerProvidedFS(t *testing.T) {
	modelEngine := &fakeModelEngine{predict: func(_ context.Context, _ State, questions []Question) (Prediction, error) {
		return fakePrediction(questions)
	}}
	engine := &fakeRuntimeEngine{fsModel: modelEngine}
	runtime, err := newRuntime(engine)
	if err != nil {
		t.Fatalf("newRuntime() returned unexpected error: %v", err)
	}
	modelFS := fstest.MapFS{"bundle.json": {Data: []byte("{}")}}
	model, err := runtime.OpenModelFS(t.Context(), modelFS, FSModelOptions{Root: ".", WorkDir: t.TempDir()})
	if err != nil {
		t.Fatalf("runtime.OpenModelFS() returned unexpected error: %v", err)
	}
	if err := model.Close(t.Context()); err != nil {
		t.Fatalf("model.Close() returned unexpected error: %v", err)
	}
	if err := runtime.Close(t.Context()); err != nil {
		t.Fatalf("runtime.Close() returned unexpected error: %v", err)
	}

	dirCalls, fsCalls, _ := engine.counts()
	if dirCalls != 0 || fsCalls != 1 {
		t.Errorf("runtime engine open calls = (dir %d, fs %d), want (0, 1)", dirCalls, fsCalls)
	}
}

func TestRuntimeRejectsInvalidOpenInputs(t *testing.T) {
	engine := &fakeRuntimeEngine{}
	runtime, err := newRuntime(engine)
	if err != nil {
		t.Fatalf("newRuntime() returned unexpected error: %v", err)
	}
	t.Cleanup(func() {
		if err := runtime.Close(context.Background()); err != nil {
			t.Errorf("runtime.Close() cleanup = %v, want nil", err)
		}
	})

	if _, err := runtime.OpenModelDir(t.Context(), "", ModelOptions{}); !errors.Is(err, ErrInvalidConfig) {
		t.Errorf("runtime.OpenModelDir(empty) error = %v, want errors.Is(_, ErrInvalidConfig)", err)
	}
	if _, err := runtime.OpenModelFS(t.Context(), nil, FSModelOptions{WorkDir: t.TempDir()}); !errors.Is(err, ErrInvalidConfig) {
		t.Errorf("runtime.OpenModelFS(nil) error = %v, want errors.Is(_, ErrInvalidConfig)", err)
	}
	if _, err := runtime.OpenModelDir(t.Context(), "/model", ModelOptions{}); !errors.Is(err, ErrNativeFailure) {
		t.Errorf("runtime.OpenModelDir() with nil engine model error = %v, want errors.Is(_, ErrNativeFailure)", err)
	}
}

func TestRuntimeConcurrentCloseRunsEngineCleanupOnce(t *testing.T) {
	started := make(chan struct{}, 1)
	release := make(chan struct{})
	engine := &fakeRuntimeEngine{close: func(ctx context.Context) error {
		started <- struct{}{}
		select {
		case <-release:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	}}
	runtime, err := newRuntime(engine)
	if err != nil {
		t.Fatalf("newRuntime() returned unexpected error: %v", err)
	}

	done := make(chan error, 2)
	go func() { done <- runtime.Close(t.Context()) }()
	<-started
	go func() { done <- runtime.Close(t.Context()) }()
	close(release)
	for range 2 {
		if err := <-done; err != nil {
			t.Errorf("concurrent runtime.Close() = %v, want nil", err)
		}
	}
	_, _, closeCalls := engine.counts()
	if closeCalls != 1 {
		t.Errorf("runtime engine close calls = %d, want 1", closeCalls)
	}
}

func TestRuntimeRefusesCloseWhileModelOpenIsInProgress(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	fakeModel := &fakeModelEngine{predict: func(_ context.Context, _ State, questions []Question) (Prediction, error) {
		return fakePrediction(questions)
	}}
	engine := &fakeRuntimeEngine{dirOpen: func(context.Context, string, ModelOptions) (modelEngine, error) {
		close(started)
		<-release
		return fakeModel, nil
	}}
	runtimeOwner, err := newRuntime(engine)
	if err != nil {
		t.Fatalf("newRuntime() returned unexpected error: %v", err)
	}

	opened := make(chan *Model, 1)
	openErr := make(chan error, 1)
	go func() {
		model, err := runtimeOwner.OpenModelDir(t.Context(), "/model", ModelOptions{})
		opened <- model
		openErr <- err
	}()
	<-started

	if err := runtimeOwner.Close(t.Context()); !errors.Is(err, ErrModelsOpen) {
		t.Errorf("runtime.Close() during open error = %v, want errors.Is(_, ErrModelsOpen)", err)
	}
	close(release)
	model := <-opened
	if err := <-openErr; err != nil {
		t.Fatalf("runtime.OpenModelDir() returned unexpected error: %v", err)
	}
	if model == nil {
		t.Fatal("runtime.OpenModelDir() model = nil, want non-nil")
	}
	if err := model.Close(t.Context()); err != nil {
		t.Fatalf("model.Close() returned unexpected error: %v", err)
	}
	if err := runtimeOwner.Close(t.Context()); err != nil {
		t.Fatalf("runtime.Close() returned unexpected error: %v", err)
	}
}

func TestRuntimeCloseCanRetryAfterEngineError(t *testing.T) {
	closeAttempt := 0
	engine := &fakeRuntimeEngine{close: func(context.Context) error {
		closeAttempt++
		if closeAttempt == 1 {
			return context.DeadlineExceeded
		}
		return nil
	}}
	runtimeOwner, err := newRuntime(engine)
	if err != nil {
		t.Fatalf("newRuntime() returned unexpected error: %v", err)
	}

	if err := runtimeOwner.Close(t.Context()); !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("first runtime.Close() error = %v, want errors.Is(_, context.DeadlineExceeded)", err)
	}
	if err := runtimeOwner.Close(t.Context()); err != nil {
		t.Fatalf("second runtime.Close() returned unexpected error: %v", err)
	}
	_, _, closeCalls := engine.counts()
	if closeCalls != 2 {
		t.Errorf("runtime engine close calls = %d, want 2", closeCalls)
	}
}

func TestRuntimePreservesModelOpenErrors(t *testing.T) {
	engine := &fakeRuntimeEngine{dirOpen: func(context.Context, string, ModelOptions) (modelEngine, error) {
		return nil, ErrInvalidBundle
	}}
	runtimeOwner, err := newRuntime(engine)
	if err != nil {
		t.Fatalf("newRuntime() returned unexpected error: %v", err)
	}
	t.Cleanup(func() {
		if err := runtimeOwner.Close(context.Background()); err != nil {
			t.Errorf("runtime.Close() cleanup = %v, want nil", err)
		}
	})

	if _, err := runtimeOwner.OpenModelDir(t.Context(), "/model", ModelOptions{}); !errors.Is(err, ErrInvalidBundle) {
		t.Errorf("runtime.OpenModelDir() error = %v, want errors.Is(_, ErrInvalidBundle)", err)
	}
	canceled, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := runtimeOwner.OpenModelFS(canceled, fstest.MapFS{}, FSModelOptions{WorkDir: t.TempDir()}); !errors.Is(err, context.Canceled) {
		t.Errorf("runtime.OpenModelFS(canceled) error = %v, want errors.Is(_, context.Canceled)", err)
	}
	dirCalls, fsCalls, _ := engine.counts()
	if dirCalls != 1 || fsCalls != 0 {
		t.Errorf("runtime engine open calls = (dir %d, fs %d), want (1, 0)", dirCalls, fsCalls)
	}
}
