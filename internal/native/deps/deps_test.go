package deps

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"testing"
)

func TestManifestIdentity(t *testing.T) {
	decoder := json.NewDecoder(bytes.NewReader(Manifest))
	decoder.DisallowUnknownFields()
	var document struct {
		SchemaVersion int `json:"schema_version"`
		Candidate     struct {
			GoVersion string `json:"go_version"`
			GOOS      string `json:"goos"`
			GOARCH    string `json:"goarch"`
			CGO       bool   `json:"cgo"`
		} `json:"candidate"`
		ONNXRuntimeGo json.RawMessage `json:"onnxruntime_go"`
		ONNXRuntime   json.RawMessage `json:"onnxruntime"`
		TokenizersGo  json.RawMessage `json:"tokenizers_go"`
		Tokenizers    json.RawMessage `json:"tokenizers_native"`
	}
	if err := decoder.Decode(&document); err != nil {
		t.Fatalf("decode native dependency manifest: %v", err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		t.Fatalf("native dependency manifest trailer: %v", err)
	}
	if document.SchemaVersion != 1 || document.Candidate.GoVersion != "1.26.6" ||
		document.Candidate.GOOS != "linux" || document.Candidate.GOARCH != "amd64" ||
		!document.Candidate.CGO {
		t.Fatalf("unexpected native candidate: %+v", document.Candidate)
	}
	for name, value := range map[string]json.RawMessage{
		"onnxruntime_go":    document.ONNXRuntimeGo,
		"onnxruntime":       document.ONNXRuntime,
		"tokenizers_go":     document.TokenizersGo,
		"tokenizers_native": document.Tokenizers,
	} {
		if len(value) == 0 {
			t.Errorf("native dependency manifest is missing %s", name)
		}
	}
}
