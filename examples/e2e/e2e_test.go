//go:build laya_native && cgo

package e2e_test

import (
	"bytes"
	"context"
	"os"
	"strings"
	"testing"

	"github.com/metalagman/laya-go"
	"github.com/metalagman/laya-go/adklaya"
	adkexample "github.com/metalagman/laya-go/examples/adk"
	"github.com/metalagman/laya-go/examples/directory"
	"github.com/metalagman/laya-go/examples/embedded"
)

const bundleID = "sha256:963bc035d885e0463ea1d7d54906cadf0e2a3a41073e131921800ea5cc358935"

func TestProtectedExamplesEndToEnd(t *testing.T) {
	bundleDir := os.Getenv("LAYA_BUNDLE_DIR")
	if bundleDir == "" || os.Getenv("LAYA_ONNXRUNTIME_LIBRARY") == "" {
		t.Skip("protected native artifacts were not supplied")
	}
	const input = "private end-to-end input"
	state := laya.TextState(input)

	t.Run("directory", func(t *testing.T) {
		var output bytes.Buffer
		if err := directory.Run(t.Context(), bundleDir, state, &output); err != nil {
			t.Fatalf("directory.Run() error = %v", err)
		}
		assertSafeOutput(t, output.String(), input)
	})

	t.Run("filesystem", func(t *testing.T) {
		var output bytes.Buffer
		if err := embedded.Run(t.Context(), os.DirFS(bundleDir), t.TempDir(), state, &output); err != nil {
			t.Fatalf("embedded.Run() error = %v", err)
		}
		assertSafeOutput(t, output.String(), input)
	})

	t.Run("ADK shared model", func(t *testing.T) {
		runtime, err := laya.NewRuntime(t.Context(), laya.RuntimeOptions{})
		if err != nil {
			t.Fatal(err)
		}
		model, err := runtime.OpenModelDir(t.Context(), bundleDir, laya.ModelOptions{QueueCapacity: 2, Truncation: laya.TruncateOverflow})
		if err != nil {
			_ = runtime.Close(context.Background())
			t.Fatal(err)
		}
		for _, test := range []struct {
			name   string
			policy adklaya.AcceptancePolicy
			route  string
		}{
			{name: "accepted", policy: adklaya.AcceptancePolicy{MaxActionProbability: 1}, route: "accepted"},
			{name: "fallback", policy: adklaya.AcceptancePolicy{MinSelectedProbability: 1, MinConfidence: 1, MaxActionProbability: 0}, route: "fallback"},
		} {
			t.Run(test.name, func(t *testing.T) {
				var output bytes.Buffer
				if err := adkexample.Run(t.Context(), model, input, test.policy, &output); err != nil {
					t.Fatalf("adk.Run() error = %v", err)
				}
				assertSafeOutput(t, output.String(), input)
				if !strings.Contains(output.String(), `"route":"`+test.route+`"`) {
					t.Errorf("output = %q, want route %q", output.String(), test.route)
				}
			})
		}
		if err := model.Close(context.Background()); err != nil {
			t.Errorf("Model.Close() error = %v", err)
		}
		if err := runtime.Close(context.Background()); err != nil {
			t.Errorf("Runtime.Close() error = %v", err)
		}
	})
}

func assertSafeOutput(t *testing.T, output, input string) {
	t.Helper()
	if !strings.Contains(output, bundleID) {
		t.Errorf("output = %q, want accepted BundleID", output)
	}
	if strings.Contains(output, input) {
		t.Errorf("output leaked raw input %q", input)
	}
}
