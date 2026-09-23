// Command embedprobe runs a protected native test using a compile-time embed.FS.
// It copies an already-local verified bundle into a private temporary Go module;
// neither the bundle nor generated binaries are added to the repository.
package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/metalagman/laya-go/internal/bundle"
)

func main() {
	if err := run(context.Background()); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(ctx context.Context) (returnErr error) {
	bundleDir := os.Getenv("LAYA_BUNDLE_DIR")
	workDir := os.Getenv("LAYA_EMBED_WORKDIR")
	if !filepath.IsAbs(bundleDir) || !filepath.IsAbs(workDir) {
		return errors.New("LAYA_BUNDLE_DIR and LAYA_EMBED_WORKDIR must be absolute existing directories")
	}
	if _, err := bundle.Verify(ctx, os.DirFS(bundleDir), "."); err != nil {
		return fmt.Errorf("verify supplied bundle: %w", err)
	}
	info, err := os.Stat(workDir)
	if err != nil {
		return fmt.Errorf("inspect embedded probe work directory: %w", err)
	}
	if !info.IsDir() {
		return errors.New("embedded probe work path is not a directory")
	}
	root, err := os.MkdirTemp(workDir, "laya-embedprobe-")
	if err != nil {
		return fmt.Errorf("create private embedded probe directory: %w", err)
	}
	defer func() { returnErr = errors.Join(returnErr, os.RemoveAll(root)) }()
	if err := copyBundle(bundleDir, filepath.Join(root, "bundle")); err != nil {
		return err
	}
	repository, err := os.Getwd()
	if err != nil {
		return err
	}
	if _, err := os.Stat(filepath.Join(repository, "go.mod")); err != nil {
		return fmt.Errorf("run embedprobe from the repository root: %w", err)
	}
	module := fmt.Sprintf("module example.com/laya-embedprobe\n\ngo 1.26.6\n\nrequire github.com/metalagman/laya-go v0.0.0\n\nreplace github.com/metalagman/laya-go => %s\n", repository)
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte(module), 0o600); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(root, "embedded_test.go"), []byte(probeTest), 0o600); err != nil {
		return err
	}
	arguments := []string{"test", "-mod=mod", "-tags=laya_native", "-count=1", "-v", "./..."}
	if os.Getenv("LAYA_EMBED_RACE") == "1" {
		arguments = append([]string{"test", "-race"}, arguments[1:]...)
	}
	command := exec.CommandContext(ctx, "go", arguments...)
	command.Dir = root
	command.Env = append(os.Environ(), "GOPROXY=off", "GOSUMDB=off", "GOWORK=off")
	command.Stdout = os.Stdout
	command.Stderr = os.Stderr
	if err := command.Run(); err != nil {
		return fmt.Errorf("run native embed.FS probe: %w", err)
	}
	return nil
}

func copyBundle(source, destination string) error {
	if err := os.Mkdir(destination, 0o700); err != nil {
		return err
	}
	return filepath.WalkDir(source, func(path string, entry fs.DirEntry, walkErr error) (returnErr error) {
		if walkErr != nil {
			return walkErr
		}
		relative, err := filepath.Rel(source, path)
		if err != nil {
			return err
		}
		if relative == "." {
			return nil
		}
		target := filepath.Join(destination, relative)
		if entry.IsDir() {
			return os.Mkdir(target, 0o700)
		}
		if !entry.Type().IsRegular() || strings.HasPrefix(entry.Name(), ".") {
			return fmt.Errorf("embedded probe requires ordinary non-hidden bundle files: %q", relative)
		}
		input, err := os.Open(path)
		if err != nil {
			return err
		}
		defer func() { returnErr = errors.Join(returnErr, input.Close()) }()
		output, err := os.OpenFile(target, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
		if err != nil {
			return err
		}
		_, copyErr := io.Copy(output, input)
		return errors.Join(copyErr, output.Close())
	})
}

const probeTest = `package embedprobe

import (
	"embed"
	"errors"
	"io/fs"
	"os"
	"reflect"
	"testing"
	"testing/fstest"

	"github.com/metalagman/laya-go"
)

//go:embed bundle
var embeddedBundle embed.FS

type alteredFS struct {
	fs.FS
	missing bool
}

func (a alteredFS) Open(name string) (fs.File, error) {
	if name == "bundle/manifest.json" {
		if a.missing {
			return nil, fs.ErrNotExist
		}
		return fstest.MapFS{"manifest.json": &fstest.MapFile{Data: []byte("{")}}.Open("manifest.json")
	}
	return a.FS.Open(name)
}

func TestRealEmbeddedInference(t *testing.T) {
	ctx := t.Context()
	runtime, err := laya.NewRuntime(ctx, laya.RuntimeOptions{})
	if err != nil { t.Fatal(err) }
	question, err := laya.NewNoulQuestion("embedded", "Is this local?", "no", "yes")
	if err != nil { t.Fatal(err) }
	directory, err := runtime.OpenModelDir(ctx, os.Getenv("LAYA_BUNDLE_DIR"), laya.ModelOptions{})
	if err != nil { t.Fatal(err) }
	want, err := directory.Predict(ctx, laya.TextState("offline embedded parity"), []laya.Question{question})
	if err != nil { t.Fatal(err) }
	if err := directory.Close(ctx); err != nil { t.Fatal(err) }
	embedded, err := runtime.OpenModelFS(ctx, embeddedBundle, laya.FSModelOptions{Root: "bundle", WorkDir: t.TempDir()})
	if err != nil { t.Fatal(err) }
	got, err := embedded.Predict(ctx, laya.TextState("offline embedded parity"), []laya.Question{question})
	if err != nil { t.Fatal(err) }
	if got.Metadata().BundleID != want.Metadata().BundleID || !reflect.DeepEqual(got.Results(), want.Results()) {
		t.Fatalf("embedded prediction differs from directory prediction: bundle %q vs %q", got.Metadata().BundleID, want.Metadata().BundleID)
	}
	if err := embedded.Close(ctx); err != nil { t.Fatal(err) }
	for _, test := range []struct{name string; source fs.FS}{
		{"missing manifest", alteredFS{FS: embeddedBundle, missing: true}},
		{"corrupt manifest", alteredFS{FS: embeddedBundle}},
	} {
		t.Run(test.name, func(t *testing.T) {
			model, err := runtime.OpenModelFS(ctx, test.source, laya.FSModelOptions{Root: "bundle", WorkDir: t.TempDir()})
			if model != nil { _ = model.Close(ctx); t.Fatal("invalid embedded bundle opened") }
			if !errors.Is(err, laya.ErrInvalidBundle) { t.Fatalf("invalid embedded bundle error = %v", err) }
		})
	}
	if err := runtime.Close(ctx); err != nil { t.Fatal(err) }
}
`
