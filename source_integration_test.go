//go:build laya_native && cgo && linux

package laya

import (
	"context"
	"errors"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sync"
	"testing"

	internalbundle "github.com/metalagman/laya-go/internal/bundle"
)

func TestProtectedSourceModeParity(t *testing.T) {
	bundleDir := protectedSourceBundle(t)
	snapshotDir := createProtectedSnapshot(t, bundleDir)
	workDir := t.TempDir()
	tests := []struct {
		name string
		open func(*Runtime) (*Model, error)
	}{
		{
			name: "ordinary directory",
			open: func(runtime *Runtime) (*Model, error) {
				return runtime.OpenModelDir(t.Context(), bundleDir, ModelOptions{Truncation: TruncateOverflow})
			},
		},
		{
			name: "Hugging Face snapshot",
			open: func(runtime *Runtime) (*Model, error) {
				return runtime.OpenModelDir(t.Context(), snapshotDir, ModelOptions{Truncation: TruncateOverflow})
			},
		},
		{
			name: "filesystem materialization",
			open: func(runtime *Runtime) (*Model, error) {
				return runtime.OpenModelFS(t.Context(), os.DirFS(bundleDir), FSModelOptions{
					WorkDir:      workDir,
					ModelOptions: ModelOptions{Truncation: TruncateOverflow},
				})
			},
		},
		{
			name: "filesystem retained reuse",
			open: func(runtime *Runtime) (*Model, error) {
				return runtime.OpenModelFS(t.Context(), os.DirFS(bundleDir), FSModelOptions{
					WorkDir:      workDir,
					ModelOptions: ModelOptions{Truncation: TruncateOverflow},
				})
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			runProtectedModelParity(t, test.open)
		})
	}
	assertOneProtectedPublication(t, workDir)
}

func TestProtectedSourceCancellationRecovery(t *testing.T) {
	bundleDir := protectedSourceBundle(t)
	workDir := t.TempDir()
	ctx, cancel := context.WithCancel(t.Context())
	fileSystem := &cancelOnSecondOpenFS{
		FS:     os.DirFS(bundleDir),
		target: "model.onnx.data",
		cancel: cancel,
	}
	runtime, err := NewRuntime(t.Context(), RuntimeOptions{})
	if err != nil {
		t.Fatalf("NewRuntime returned unexpected error: %v", err)
	}
	model, err := runtime.OpenModelFS(ctx, fileSystem, FSModelOptions{WorkDir: workDir})
	if model != nil {
		_ = model.Close(t.Context())
		t.Fatal("OpenModelFS(canceled copy) returned a non-nil Model")
	}
	if !errors.Is(err, context.Canceled) {
		_ = runtime.Close(t.Context())
		t.Fatalf("OpenModelFS(canceled copy) error = %v, want context.Canceled", err)
	}
	assertProtectedStagingEmpty(t, workDir)

	model, err = runtime.OpenModelFS(t.Context(), os.DirFS(bundleDir), FSModelOptions{WorkDir: workDir})
	if err != nil {
		_ = runtime.Close(t.Context())
		t.Fatalf("OpenModelFS(after cancellation) returned unexpected error: %v", err)
	}
	question, err := NewNoulQuestion("recovery", "Is recovery usable?", "no", "yes")
	if err != nil {
		t.Fatalf("NewNoulQuestion returned unexpected error: %v", err)
	}
	if _, err := model.Predict(t.Context(), TextState("local source recovery"), []Question{question}); err != nil {
		t.Errorf("Predict(after cancellation) returned unexpected error: %v", err)
	}
	if err := model.Close(t.Context()); err != nil {
		t.Errorf("Model.Close(after cancellation) returned unexpected error: %v", err)
	}
	if err := runtime.Close(t.Context()); err != nil {
		t.Errorf("Runtime.Close(after cancellation) returned unexpected error: %v", err)
	}
	assertOneProtectedPublication(t, workDir)
}

func TestProtectedSourceSharedPublicationLifecycle(t *testing.T) {
	bundleDir := protectedSourceBundle(t)
	workDir := t.TempDir()
	runtime, err := NewRuntime(t.Context(), RuntimeOptions{})
	if err != nil {
		t.Fatalf("NewRuntime returned unexpected error: %v", err)
	}
	const modelCount = 2
	models := make(chan *Model, modelCount)
	errorsByOpen := make(chan error, modelCount)
	start := make(chan struct{})
	var group sync.WaitGroup
	for range modelCount {
		group.Add(1)
		go func() {
			defer group.Done()
			<-start
			model, openErr := runtime.OpenModelFS(t.Context(), os.DirFS(bundleDir), FSModelOptions{WorkDir: workDir})
			if openErr == nil {
				models <- model
			}
			errorsByOpen <- openErr
		}()
	}
	close(start)
	group.Wait()
	close(models)
	close(errorsByOpen)
	for openErr := range errorsByOpen {
		if openErr != nil {
			t.Fatalf("concurrent OpenModelFS returned unexpected error: %v", openErr)
		}
	}
	opened := make([]*Model, 0, modelCount)
	for model := range models {
		opened = append(opened, model)
	}
	if len(opened) != modelCount {
		t.Fatalf("opened Models = %d, want %d", len(opened), modelCount)
	}
	question, err := NewNoulQuestion("shared", "Is the shared publication usable?", "no", "yes")
	if err != nil {
		t.Fatalf("NewNoulQuestion returned unexpected error: %v", err)
	}
	if err := opened[1].Close(t.Context()); err != nil {
		t.Fatalf("second Model.Close returned unexpected error: %v", err)
	}
	if _, err := opened[0].Predict(t.Context(), TextState("first lease remains live"), []Question{question}); err != nil {
		t.Errorf("Predict after peer close returned unexpected error: %v", err)
	}
	if err := opened[0].Close(t.Context()); err != nil {
		t.Errorf("first Model.Close returned unexpected error: %v", err)
	}
	if err := runtime.Close(t.Context()); err != nil {
		t.Errorf("Runtime.Close returned unexpected error: %v", err)
	}
	assertOneProtectedPublication(t, workDir)
	assertProtectedStagingEmpty(t, workDir)
}

func protectedSourceBundle(t *testing.T) string {
	t.Helper()
	bundleDir := os.Getenv("LAYA_BUNDLE_DIR")
	if bundleDir == "" || os.Getenv(nativeLibraryEnvironment) == "" {
		t.Skip("protected native artifacts were not supplied")
	}
	return bundleDir
}

func createProtectedSnapshot(t *testing.T, bundleDir string) string {
	t.Helper()
	manifestBytes, err := os.ReadFile(filepath.Join(bundleDir, "manifest.json"))
	if err != nil {
		t.Fatalf("ReadFile(protected manifest): %v", err)
	}
	manifest, err := internalbundle.Parse(manifestBytes)
	if err != nil {
		t.Fatalf("bundle.Parse(protected manifest): %v", err)
	}
	temporary := t.TempDir()
	repository := filepath.Join(temporary, "models--convaiinnovations--laya-multilingual")
	blobs := filepath.Join(repository, "blobs")
	snapshot := filepath.Join(repository, "snapshots", manifest.Provenance.SourceModel.Revision)
	if err := os.MkdirAll(blobs, 0o700); err != nil {
		t.Fatalf("MkdirAll(blobs): %v", err)
	}
	if err := os.MkdirAll(snapshot, 0o700); err != nil {
		t.Fatalf("MkdirAll(snapshot): %v", err)
	}
	if err := os.WriteFile(filepath.Join(snapshot, "manifest.json"), manifestBytes, 0o600); err != nil {
		t.Fatalf("WriteFile(snapshot manifest): %v", err)
	}
	for _, artifact := range manifest.Files {
		blob := filepath.Join(blobs, artifact.SHA256)
		linkOrCopyProtectedBlob(t, filepath.Join(bundleDir, filepath.FromSlash(artifact.Path)), blob)
		link := filepath.Join(snapshot, filepath.FromSlash(artifact.Path))
		if err := os.MkdirAll(filepath.Dir(link), 0o700); err != nil {
			t.Fatalf("MkdirAll(snapshot artifact): %v", err)
		}
		target, err := filepath.Rel(filepath.Dir(link), blob)
		if err != nil {
			t.Fatalf("Rel(snapshot blob): %v", err)
		}
		if err := os.Symlink(target, link); err != nil {
			t.Fatalf("Symlink(snapshot artifact %q): %v", artifact.Path, err)
		}
	}
	return snapshot
}

func linkOrCopyProtectedBlob(t *testing.T, source, destination string) {
	t.Helper()
	if err := os.Link(source, destination); err == nil || errors.Is(err, fs.ErrExist) {
		return
	}
	input, err := os.Open(source)
	if err != nil {
		t.Fatalf("open protected blob source: %v", err)
	}
	defer func() {
		if err := input.Close(); err != nil {
			t.Errorf("close protected blob source: %v", err)
		}
	}()
	output, err := os.OpenFile(destination, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if errors.Is(err, fs.ErrExist) {
		return
	}
	if err != nil {
		t.Fatalf("create protected blob copy: %v", err)
	}
	buffer := make([]byte, 64*1024)
	if _, err := io.CopyBuffer(output, input, buffer); err != nil {
		_ = output.Close()
		t.Fatalf("copy protected blob: %v", err)
	}
	if err := output.Sync(); err != nil {
		_ = output.Close()
		t.Fatalf("sync protected blob: %v", err)
	}
	if err := output.Close(); err != nil {
		t.Fatalf("close protected blob copy: %v", err)
	}
}

func assertOneProtectedPublication(t *testing.T, workDir string) {
	t.Helper()
	bundles := filepath.Join(workDir, ".laya-go", "bundles")
	entries, err := os.ReadDir(bundles)
	if err != nil {
		t.Fatalf("ReadDir(materialized bundles): %v", err)
	}
	if len(entries) != 1 || !entries[0].IsDir() {
		t.Fatalf("materialized bundle entries = %v, want one directory", protectedEntryNames(entries))
	}
	manifest, err := internalbundle.Verify(t.Context(), os.DirFS(filepath.Join(bundles, entries[0].Name())), ".")
	if err != nil {
		t.Fatalf("verify retained publication: %v", err)
	}
	wantBundleID := string(loadTransformCorpus(t).Metadata.BundleID)
	if manifest.ID() != wantBundleID {
		t.Errorf("retained publication BundleID = %q, want %q", manifest.ID(), wantBundleID)
	}
}

func assertProtectedStagingEmpty(t *testing.T, workDir string) {
	t.Helper()
	entries, err := os.ReadDir(filepath.Join(workDir, ".laya-go", "staging"))
	if err != nil {
		t.Fatalf("ReadDir(materialization staging): %v", err)
	}
	if len(entries) != 0 {
		t.Errorf("materialization staging entries = %v, want empty", protectedEntryNames(entries))
	}
}

func protectedEntryNames(entries []os.DirEntry) []string {
	names := make([]string, len(entries))
	for index, entry := range entries {
		names[index] = entry.Name()
	}
	return names
}

type cancelOnSecondOpenFS struct {
	fs.FS
	mu     sync.Mutex
	target string
	opens  int
	cancel context.CancelFunc
}

func (f *cancelOnSecondOpenFS) Open(name string) (fs.File, error) {
	file, err := f.FS.Open(name)
	if err != nil || name != f.target {
		return file, err
	}
	f.mu.Lock()
	f.opens++
	shouldCancel := f.opens == 2
	f.mu.Unlock()
	if !shouldCancel {
		return file, nil
	}
	return &cancelOnReadFile{File: file, cancel: f.cancel}, nil
}

type cancelOnReadFile struct {
	fs.File
	once   sync.Once
	cancel context.CancelFunc
}

func (f *cancelOnReadFile) Read(data []byte) (int, error) {
	count, err := f.File.Read(data)
	f.once.Do(f.cancel)
	return count, err
}
