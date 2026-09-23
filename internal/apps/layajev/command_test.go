package layajev

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) { return f(request) }

func TestCommandHelp(t *testing.T) {
	t.Parallel()
	command := NewCommand()
	var output bytes.Buffer
	command.SetOut(&output)
	command.SetErr(&output)
	command.SetArgs([]string{"--help"})
	if err := command.ExecuteContext(context.Background()); err != nil {
		t.Fatalf("Execute help: %v", err)
	}
	for _, subcommand := range []string{"serve", "doctor", "fetch", "convert"} {
		if !strings.Contains(output.String(), subcommand) {
			t.Errorf("help missing %q: %s", subcommand, output.String())
		}
	}
}

func TestPackagedONNXRuntimeLibrary(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	executable := filepath.Join(dir, "layajev")
	library := filepath.Join(dir, "libonnxruntime.so.1.29.0")
	if got := packagedONNXRuntimeLibrary(executable); got != "" {
		t.Errorf("missing packaged library = %q, want empty", got)
	}
	if err := os.Mkdir(library, 0o755); err != nil {
		t.Fatalf("create library directory: %v", err)
	}
	if got := packagedONNXRuntimeLibrary(executable); got != "" {
		t.Errorf("directory packaged library = %q, want empty", got)
	}
	if err := os.Remove(library); err != nil {
		t.Fatalf("remove library directory: %v", err)
	}
	if err := os.WriteFile(library, []byte("test library"), 0o644); err != nil {
		t.Fatalf("create library file: %v", err)
	}
	if got := packagedONNXRuntimeLibrary(executable); got != library {
		t.Errorf("packaged library = %q, want %q", got, library)
	}
}

func TestFetchPinned(t *testing.T) {
	t.Parallel()
	content := []byte("official pinned artifact")
	digest := sha256.Sum256(content)
	profile := pinnedProfile{SourceFiles: []sourceFile{{Path: "tokenizer/config.json", Size: int64(len(content)), SHA256: hex.EncodeToString(digest[:])}}}
	profile.SourceModel.ID = officialModelID
	profile.SourceModel.Revision = officialRevision
	client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if !strings.HasSuffix(r.URL.Path, "/tokenizer/config.json") {
			t.Errorf("request path = %q", r.URL.Path)
		}
		return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(bytes.NewReader(content))}, nil
	})}
	destination := filepath.Join(t.TempDir(), "snapshot")
	if err := fetchPinnedFrom(context.Background(), profile, destination, client, "https://example.test"); err != nil {
		t.Fatalf("fetchPinnedFrom: %v", err)
	}
	got, err := os.ReadFile(filepath.Join(destination, "tokenizer", "config.json"))
	if err != nil || !bytes.Equal(got, content) {
		t.Fatalf("fetched artifact = %q, error %v", got, err)
	}
	if err := fetchPinnedFrom(context.Background(), profile, destination, client, "https://example.test"); err == nil {
		t.Error("fetch into existing destination succeeded")
	}
}

func TestFetchRejectsChecksumMismatch(t *testing.T) {
	t.Parallel()
	profile := pinnedProfile{SourceFiles: []sourceFile{{Path: "file.txt", Size: 3, SHA256: strings.Repeat("0", 64)}}}
	profile.SourceModel.ID = officialModelID
	profile.SourceModel.Revision = officialRevision
	client := &http.Client{Transport: roundTripFunc(func(_ *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader("abc"))}, nil
	})}
	destination := filepath.Join(t.TempDir(), "snapshot")
	if err := fetchPinnedFrom(context.Background(), profile, destination, client, "https://example.test"); err == nil || !strings.Contains(err.Error(), "SHA-256 mismatch") {
		t.Errorf("fetch error = %v, want SHA-256 mismatch", err)
	}
	if _, err := os.Lstat(destination); !os.IsNotExist(err) {
		t.Errorf("destination exists after failed verification: %v", err)
	}
}

func TestConvertValidation(t *testing.T) {
	t.Parallel()
	if err := convert(context.Background(), t.TempDir(), "", "", "", 0); err == nil {
		t.Error("convert accepted empty inputs")
	}
}

func TestLoadPinnedProfile(t *testing.T) {
	t.Parallel()
	profile, err := loadPinnedProfile("")
	if err != nil {
		t.Fatalf("loadPinnedProfile: %v", err)
	}
	if profile.SourceModel.ID != officialModelID || profile.SourceModel.Revision != officialRevision || len(profile.SourceFiles) == 0 {
		t.Errorf("unexpected pinned source: %+v", profile.SourceModel)
	}
	data, err := os.ReadFile("../../../tools/export/profiles/laya-multilingual-v1.json")
	if err != nil {
		t.Fatalf("read canonical export profile: %v", err)
	}
	var canonical pinnedProfile
	if err := json.Unmarshal(data, &canonical); err != nil {
		t.Fatalf("parse canonical export profile: %v", err)
	}
	if !reflect.DeepEqual(profile, canonical) {
		t.Error("embedded fetch source differs from canonical export profile")
	}
	fromCheckout, err := loadPinnedProfile(filepath.Join("..", "..", ".."))
	if err != nil {
		t.Fatalf("load checkout profile: %v", err)
	}
	if !reflect.DeepEqual(profile, fromCheckout) {
		t.Error("embedded fetch source differs from checkout profile")
	}
}

func TestFetchStandalone(t *testing.T) {
	if os.Getenv("LAYAJEV_TEST_STANDALONE") == "1" {
		command := NewCommand()
		command.SetArgs([]string{"fetch", "--destination", "."})
		err := command.ExecuteContext(context.Background())
		if err == nil || !strings.Contains(err.Error(), "already exists") {
			t.Fatalf("fetch outside checkout error = %v, want existing destination", err)
		}
		return
	}
	command := exec.Command(os.Args[0], "-test.run=^TestFetchStandalone$")
	command.Dir = t.TempDir()
	command.Env = append(os.Environ(), "LAYAJEV_TEST_STANDALONE=1")
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("fetch outside checkout: %v\n%s", err, output)
	}
}
