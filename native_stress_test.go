//go:build laya_native && laya_native_diagnostics && cgo && linux

package laya

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"sync"
	"testing"
)

func TestProtectedNativeLifecycleStress(t *testing.T) {
	bundleDir := os.Getenv("LAYA_BUNDLE_DIR")
	if bundleDir == "" || os.Getenv(nativeLibraryEnvironment) == "" {
		t.Skip("protected native artifacts were not supplied")
	}
	trimNativeAllocatorForTest()
	baselineRSS := processRSSBytes(t)

	layaRuntime, err := NewRuntime(t.Context(), RuntimeOptions{})
	if err != nil {
		t.Fatalf("NewRuntime returned unexpected error: %v", err)
	}
	t.Logf("RSS after runtime=%d MiB", processRSSBytes(t)>>20)
	first, err := layaRuntime.OpenModelDir(t.Context(), bundleDir, ModelOptions{})
	if err != nil {
		t.Fatalf("open first model: %v", err)
	}
	t.Logf("RSS after first model=%d MiB", processRSSBytes(t)>>20)
	second, err := layaRuntime.OpenModelDir(t.Context(), bundleDir, ModelOptions{})
	if err != nil {
		_ = first.Close(t.Context())
		_ = layaRuntime.Close(t.Context())
		t.Fatalf("open second model: %v", err)
	}
	t.Logf("RSS after second model=%d MiB", processRSSBytes(t)>>20)
	if err := layaRuntime.Close(t.Context()); !errors.Is(err, ErrModelsOpen) {
		t.Errorf("runtime Close with live models = %v, want ErrModelsOpen", err)
	}

	question, err := NewNoulQuestion("stress", "Is the route safe?", "unsafe", "safe")
	if err != nil {
		t.Fatalf("construct stress question: %v", err)
	}
	models := []*Model{first, second}
	start := make(chan struct{})
	errorsByModel := make(chan error, len(models))
	var group sync.WaitGroup
	for _, model := range models {
		group.Add(1)
		go func(model *Model) {
			defer group.Done()
			<-start
			_, predictErr := model.Predict(t.Context(), TextState("local stress state"), []Question{question})
			errorsByModel <- predictErr
		}(model)
	}
	close(start)
	group.Wait()
	t.Logf("RSS after concurrent predictions=%d MiB", processRSSBytes(t)>>20)
	close(errorsByModel)
	for predictErr := range errorsByModel {
		if predictErr != nil {
			t.Errorf("concurrent model Predict returned unexpected error: %v", predictErr)
		}
	}

	if err := first.Close(t.Context()); err != nil {
		t.Fatalf("close first model: %v", err)
	}
	t.Logf("RSS after first close=%d MiB", processRSSBytes(t)>>20)
	if _, err := second.Predict(t.Context(), TextState("second remains live"), []Question{question}); err != nil {
		t.Fatalf("second model after first close: %v", err)
	}
	if err := second.Close(t.Context()); err != nil {
		t.Fatalf("close second model: %v", err)
	}
	t.Logf("RSS after second close=%d MiB", processRSSBytes(t)>>20)
	if err := layaRuntime.Close(t.Context()); err != nil {
		t.Fatalf("close runtime: %v", err)
	}
	t.Logf("RSS after runtime close=%d MiB", processRSSBytes(t)>>20)

	runtime.GC()
	debug.FreeOSMemory()
	trimNativeAllocatorForTest()
	finalRSS := processRSSBytes(t)
	const retainedRSSLimit = int64(128 << 20)
	if retained := finalRSS - baselineRSS; retained > retainedRSSLimit {
		t.Errorf(
			"retained RSS after complete native teardown = %d MiB, limit %d MiB",
			retained>>20,
			retainedRSSLimit>>20,
		)
	}
	t.Logf("native lifecycle RSS baseline=%d MiB final=%d MiB", baselineRSS>>20, finalRSS>>20)
}

func processRSSBytes(t *testing.T) int64 {
	t.Helper()
	data, err := os.ReadFile("/proc/self/status")
	if err != nil {
		t.Fatalf("read process status: %v", err)
	}
	for line := range strings.SplitSeq(string(data), "\n") {
		fields := strings.Fields(line)
		if len(fields) != 3 || fields[0] != "VmRSS:" || fields[2] != "kB" {
			continue
		}
		kilobytes, parseErr := strconv.ParseInt(fields[1], 10, 64)
		if parseErr != nil {
			t.Fatalf("parse process RSS: %v", parseErr)
		}
		return kilobytes << 10
	}
	t.Fatal(fmt.Errorf("VmRSS is absent from /proc/self/status"))
	return 0
}
