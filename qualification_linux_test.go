//go:build linux && amd64 && laya_native && cgo

package laya

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/metalagman/laya-go/internal/bundle"
)

func TestProtectedQualificationMetrics(t *testing.T) {
	bundleDir := os.Getenv("LAYA_BUNDLE_DIR")
	if bundleDir == "" || os.Getenv(nativeLibraryEnvironment) == "" {
		t.Skip("protected native artifacts were not supplied")
	}
	manifestBytes, err := os.ReadFile(filepath.Join(bundleDir, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	manifest, err := bundle.Parse(manifestBytes)
	if err != nil {
		t.Fatal(err)
	}
	host, err := os.Hostname()
	if err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	r, err := NewRuntime(t.Context(), RuntimeOptions{})
	if err != nil {
		t.Fatal(err)
	}
	runtimeOpen := time.Since(start)
	start = time.Now()
	model, err := r.OpenModelDir(t.Context(), bundleDir, ModelOptions{QueueCapacity: 4, Truncation: TruncateOverflow})
	if err != nil {
		_ = r.Close(t.Context())
		t.Fatal(err)
	}
	modelOpen := time.Since(start)
	question, err := NewChoiceQuestion("route", "Pick one", []ChoiceCriterion{{ID: "left", Description: "go left"}, {ID: "right", Description: "go right"}})
	if err != nil {
		t.Fatal(err)
	}
	const runs = 20
	latencies := make([]time.Duration, 0, runs)
	for range runs {
		start = time.Now()
		prediction, err := model.Predict(t.Context(), TextState("binary route"), []Question{question})
		if err != nil {
			t.Fatal(err)
		}
		if prediction.Metadata().BundleID != BundleID("sha256:963bc035d885e0463ea1d7d54906cadf0e2a3a41073e131921800ea5cc358935") {
			t.Fatalf("BundleID = %q", prediction.Metadata().BundleID)
		}
		latencies = append(latencies, time.Since(start))
	}
	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })
	total := time.Duration(0)
	for _, latency := range latencies {
		total += latency
	}
	queueWait := measureQueueWait(t, model)
	record := fmt.Sprintf("QUALIFICATION host=%q os=linux arch=amd64 model=%q precision=%q bundle_id=%q ort_version=%q ort_sha256=%q ort_threads=default gomaxprocs=%d cpus=%d runtime_open_ms=%.3f model_open_ms=%.3f rss_mib=%d process_threads=%d queue_probe_hold_ms=25 queue_wait_ms=%.3f runs=%d p50_ms=%.3f p95_ms=%.3f sequential_throughput_per_s=%.3f", host, manifest.Provenance.SourceModel.ID, manifest.Model.Precision, manifest.ID(), nativeONNXRuntimeVersion, nativeLibrarySHA256, runtime.GOMAXPROCS(0), runtime.NumCPU(), millis(runtimeOpen), millis(modelOpen), residentMiB(t), threadCount(t), millis(queueWait), runs, millis(latencies[runs/2]), millis(latencies[(runs*95+99)/100-1]), float64(runs)/total.Seconds())
	for _, field := range []string{"host=", "model=", "precision=", "bundle_id=", "ort_version=", "ort_sha256=", "ort_threads=", "gomaxprocs=", "cpus=", "runtime_open_ms=", "model_open_ms=", "rss_mib=", "process_threads=", "queue_wait_ms=", "p50_ms=", "p95_ms=", "sequential_throughput_per_s="} {
		if !strings.Contains(record, field) {
			t.Fatalf("qualification record omits %q", field)
		}
	}
	t.Log(record)
	if err := model.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	if err := r.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
}

// measureQueueWait holds the single inference slot for a known interval, then
// measures one queued admission. It is a queue-behavior probe, not service load.
func measureQueueWait(t *testing.T, model *Model) time.Duration {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	defer cancel()
	if err := model.acquire(ctx); err != nil {
		t.Fatal(err)
	}
	type outcome struct {
		wait time.Duration
		err  error
	}
	queued := make(chan outcome, 1)
	started := make(chan struct{})
	go func() {
		start := time.Now()
		close(started)
		err := model.acquire(ctx)
		wait := time.Since(start)
		if err == nil {
			model.release()
		}
		queued <- outcome{wait: wait, err: err}
	}()
	<-started
	time.Sleep(25 * time.Millisecond)
	model.release()
	result := <-queued
	if result.err != nil {
		t.Fatal(result.err)
	}
	return result.wait
}

func millis(duration time.Duration) float64 { return float64(duration) / float64(time.Millisecond) }

func residentMiB(t *testing.T) int64 {
	t.Helper()
	data, err := os.ReadFile("/proc/self/statm")
	if err != nil {
		t.Fatal(err)
	}
	fields := strings.Fields(string(data))
	pages, err := strconv.ParseInt(fields[1], 10, 64)
	if err != nil {
		t.Fatal(err)
	}
	return pages * int64(os.Getpagesize()) / (1024 * 1024)
}

func threadCount(t *testing.T) int {
	t.Helper()
	data, err := os.ReadFile("/proc/self/status")
	if err != nil {
		t.Fatal(err)
	}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "Threads:") {
			var count int
			if _, err := fmt.Sscanf(line, "Threads: %d", &count); err != nil {
				t.Fatal(err)
			}
			return count
		}
	}
	t.Fatal("Threads field absent from /proc/self/status")
	return 0
}
