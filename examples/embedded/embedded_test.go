package embedded_test

import (
	"bytes"
	"errors"
	"testing"
	"testing/fstest"

	"github.com/metalagman/laya-go"
	"github.com/metalagman/laya-go/examples/embedded"
)

func TestRunRequiresExplicitNativeRuntime(t *testing.T) {
	t.Setenv("LAYA_ONNXRUNTIME_LIBRARY", "")
	var output bytes.Buffer
	err := embedded.Run(t.Context(), fstest.MapFS{}, t.TempDir(), laya.TextState("secret input"), &output)
	if !errors.Is(err, laya.ErrNativeUnavailable) {
		t.Fatalf("Run() error = %v, want ErrNativeUnavailable", err)
	}
	if output.Len() != 0 {
		t.Fatalf("Run() wrote output on failure: %q", output.String())
	}
}
