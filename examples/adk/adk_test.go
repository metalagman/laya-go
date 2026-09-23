package adk_test

import (
	"bytes"
	"errors"
	"testing"

	"github.com/metalagman/laya-go"
	"github.com/metalagman/laya-go/adklaya"
	adkexample "github.com/metalagman/laya-go/examples/adk"
)

func TestRunRejectsNilModelWithoutOutput(t *testing.T) {
	var output bytes.Buffer
	err := adkexample.Run(t.Context(), nil, "secret", adklaya.AcceptancePolicy{}, &output)
	if !errors.Is(err, laya.ErrInvalidConfig) {
		t.Fatalf("Run() error = %v, want ErrInvalidConfig", err)
	}
	if output.Len() != 0 {
		t.Fatalf("output = %q, want empty", output.String())
	}
}
