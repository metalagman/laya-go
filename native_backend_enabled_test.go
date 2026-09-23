//go:build laya_native && cgo

package laya

import (
	"errors"
	"testing"
)

func TestNativeTelemetryOptOutValues(t *testing.T) {
	tests := []struct {
		name  string
		value string
		want  bool
	}{
		{name: "one", value: "1", want: true},
		{name: "true", value: "true", want: true},
		{name: "case and whitespace", value: " YES ", want: true},
		{name: "on", value: "on", want: true},
		{name: "short yes", value: "y", want: true},
		{name: "empty"},
		{name: "zero", value: "0"},
		{name: "false", value: "false"},
		{name: "unknown", value: "enabled"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := nativeTelemetryDisabled(test.value); got != test.want {
				t.Errorf("nativeTelemetryDisabled(%q) = %t, want %t", test.value, got, test.want)
			}
		})
	}
}

func TestNativeRuntimeRequiresPreInitializationTelemetryOptOut(t *testing.T) {
	t.Setenv(nativeTelemetryEnvironment, "")
	_, err := NewRuntime(t.Context(), RuntimeOptions{})
	if !errors.Is(err, ErrNativeUnavailable) {
		t.Fatalf("NewRuntime error = %v, want errors.Is(_, ErrNativeUnavailable)", err)
	}
}
