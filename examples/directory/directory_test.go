package directory_test

import (
	"bytes"
	"errors"
	"testing"

	"github.com/metalagman/laya-go"
	"github.com/metalagman/laya-go/examples/directory"
)

func TestRunRequiresExplicitNativeRuntime(t *testing.T) {
	t.Setenv("LAYA_ONNXRUNTIME_LIBRARY", "")
	var output bytes.Buffer
	err := directory.Run(t.Context(), "/caller/bundle", laya.TextState("secret input"), &output)
	if !errors.Is(err, laya.ErrNativeUnavailable) {
		t.Fatalf("Run() error = %v, want ErrNativeUnavailable", err)
	}
	if output.Len() != 0 {
		t.Fatalf("Run() wrote output on failure: %q", output.String())
	}
}
