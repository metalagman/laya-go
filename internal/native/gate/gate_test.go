//go:build cgo && laya_native

package gate_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/metalagman/laya-go/internal/bundle"
	"github.com/metalagman/laya-go/internal/native/tokenizer"
	"github.com/metalagman/laya-go/internal/parity"
)

const (
	onnxRuntimeLibrarySHA256 = "5715f06d8992ca8eeeddcce43df3a7d38f97d537052126f558e912cb312460ca"
	tokenizersLibrarySHA256  = "e6862b31745bb7d07980fcee70e49cd3b4318097609180f5d2d3fb394f305d50"
)

func TestNativeCompatibilityGate(t *testing.T) {
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" {
		t.Fatalf("native compatibility gate requires linux/amd64, got %s/%s", runtime.GOOS, runtime.GOARCH)
	}

	bundleDir := requiredPath(t, "LAYA_BUNDLE_DIR")
	onnxLibrary := requiredPath(t, "LAYA_ONNXRUNTIME_LIBRARY")
	tokenizersLibrary := requiredPath(t, "LAYA_TOKENIZERS_LIBRARY")
	corpusPath := requiredPath(t, "LAYA_PARITY_CORPUS")
	requireDigest(t, onnxLibrary, onnxRuntimeLibrarySHA256)
	requireDigest(t, tokenizersLibrary, tokenizersLibrarySHA256)

	manifest, err := bundle.Verify(context.Background(), os.DirFS(bundleDir), ".")
	if err != nil {
		t.Fatalf("verify complete bundle before native load: %v", err)
	}
	if manifest.Model.Opset != 18 || manifest.Model.Precision != "fp32" {
		t.Fatalf("unexpected model contract: opset=%d precision=%q", manifest.Model.Opset, manifest.Model.Precision)
	}

	corpusBytes, err := os.ReadFile(corpusPath)
	if err != nil {
		t.Fatalf("read parity corpus: %v", err)
	}
	corpus, err := parity.Load(corpusBytes)
	if err != nil {
		t.Fatalf("load parity corpus: %v", err)
	}
	if corpus.Metadata.BundleID != manifest.ID() {
		t.Fatalf("corpus bundle ID %q does not match verified bundle %q", corpus.Metadata.BundleID, manifest.ID())
	}

	t.Run("local tokenizer", func(t *testing.T) {
		tokenizerPath := filepath.Join(bundleDir, filepath.FromSlash(manifest.Tokenizer.JSONPath))
		fromFile, err := tokenizer.FromFile(tokenizerPath)
		if err != nil {
			t.Fatalf("open local tokenizer: %v", err)
		}
		defer fromFile.Close() //nolint:errcheck // Explicit close below verifies the result.
		assertTokenizerSample(t, fromFile)
		if err := fromFile.Close(); err != nil {
			t.Fatalf("close file tokenizer: %v", err)
		}
		if err := fromFile.Close(); err != nil {
			t.Fatalf("close file tokenizer again: %v", err)
		}

		tokenizerBytes, err := os.ReadFile(tokenizerPath)
		if err != nil {
			t.Fatalf("read local tokenizer: %v", err)
		}
		fromBytes, err := tokenizer.FromBytes(tokenizerBytes)
		if err != nil {
			t.Fatalf("open tokenizer bytes: %v", err)
		}
		defer fromBytes.Close() //nolint:errcheck // Explicit close below verifies the result.
		assertTokenizerSample(t, fromBytes)
		if err := fromBytes.Close(); err != nil {
			t.Fatalf("close byte tokenizer: %v", err)
		}
		if version := tokenizer.Version(); version != "1.26.0" {
			t.Fatalf("linked tokenizer version = %q, want 1.26.0", version)
		} else {
			t.Logf("linked tokenizer version: %s", version)
		}
	})

	ort.SetSharedLibraryPath(onnxLibrary)
	for iteration := 0; iteration < 3; iteration++ {
		if err := initializeORT(); err != nil {
			t.Fatalf("initialize ONNX Runtime cycle %d: %v", iteration, err)
		}
		if got := ort.GetVersion(); got != "1.29.0" {
			t.Fatalf("ONNX Runtime version = %q, want 1.29.0", got)
		}
		if err := ort.DestroyEnvironment(); err != nil {
			t.Fatalf("destroy ONNX Runtime cycle %d: %v", iteration, err)
		}
	}

	if err := initializeORT(); err != nil {
		t.Fatalf("initialize ONNX Runtime for graph probe: %v", err)
	}
	defer func() {
		if err := ort.DestroyEnvironment(); err != nil {
			t.Errorf("destroy ONNX Runtime after graph probe: %v", err)
		}
	}()

	modelPath := filepath.Join(bundleDir, filepath.FromSlash(manifest.Model.Path))
	inputNames := tensorNames(manifest.Model.Inputs)
	outputNames := tensorNames(manifest.Model.Outputs)
	session, err := ort.NewDynamicAdvancedSession(modelPath, inputNames, outputNames, nil)
	if err != nil {
		t.Fatalf("open external-data ONNX graph: %v", err)
	}
	defer func() {
		if err := session.Destroy(); err != nil {
			t.Errorf("destroy ONNX session: %v", err)
		}
	}()

	t.Run("five input two output smoke", func(t *testing.T) {
		inputs, cleanup := makeInputs(t, corpus.Runs[0].Tensors)
		defer cleanup()
		outputs := make([]ort.Value, len(outputNames))
		if err := session.Run(inputs, outputs); err != nil {
			t.Fatalf("run accepted graph: %v", err)
		}
		defer destroyValues(t, outputs)
		verifyOutputs(t, outputs, corpus.Runs[0])
	})

	t.Run("live termination", func(t *testing.T) {
		inputs, cleanup := makeInputs(t, corpus.Runs[len(corpus.Runs)-1].Tensors)
		defer cleanup()
		outputs := make([]ort.Value, len(outputNames))
		defer destroyValues(t, outputs)
		runOptions, err := ort.NewRunOptions()
		if err != nil {
			t.Fatalf("create run options: %v", err)
		}
		defer func() {
			if err := runOptions.Destroy(); err != nil {
				t.Errorf("destroy run options: %v", err)
			}
		}()

		started := make(chan struct{})
		done := make(chan error, 1)
		go func() {
			close(started)
			done <- session.RunWithOptions(inputs, outputs, runOptions)
		}()
		<-started
		runtime.Gosched()
		if err := runOptions.Terminate(); err != nil {
			t.Fatalf("terminate active run: %v", err)
		}
		select {
		case err := <-done:
			if err == nil {
				t.Fatal("terminated run unexpectedly succeeded")
			}
			t.Logf("terminated run returned: %v", err)
		case <-time.After(30 * time.Second):
			t.Fatal("terminated run did not return within 30 seconds")
		}

		recoveryInputs, recoveryCleanup := makeInputs(t, corpus.Runs[0].Tensors)
		defer recoveryCleanup()
		recoveryOutputs := make([]ort.Value, len(outputNames))
		if err := session.Run(recoveryInputs, recoveryOutputs); err != nil {
			t.Fatalf("run after termination: %v", err)
		}
		defer destroyValues(t, recoveryOutputs)
		verifyOutputs(t, recoveryOutputs, corpus.Runs[0])
	})
}

func initializeORT() error {
	if !telemetryDisabled(os.Getenv("ORT_DISABLE_TELEMETRY")) {
		return errors.New("ORT_DISABLE_TELEMETRY must be set before native initialization")
	}
	if err := ort.InitializeEnvironment(ort.WithLogLevelError()); err != nil {
		return err
	}
	if err := ort.DisableTelemetry(); err != nil {
		return errors.Join(err, ort.DestroyEnvironment())
	}
	return nil
}

func telemetryDisabled(value string) bool {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "1", "true", "yes", "on", "y":
		return true
	default:
		return false
	}
}

func assertTokenizerSample(t *testing.T, instance *tokenizer.Tokenizer) {
	t.Helper()
	got, err := instance.EncodeErr("Choose the safest route", false)
	if err != nil {
		t.Fatalf("encode tokenizer sample: %v", err)
	}
	want := []uint32{19262, 573, 89659, 8750}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("tokenizer sample IDs = %v, want %v", got, want)
	}
}

func requiredPath(t *testing.T, name string) string {
	t.Helper()
	value := os.Getenv(name)
	if value == "" {
		t.Fatalf("%s is required", name)
	}
	absolute, err := filepath.Abs(value)
	if err != nil {
		t.Fatalf("resolve %s: %v", name, err)
	}
	return absolute
}

func requireDigest(t *testing.T, path, want string) {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatalf("open native artifact %q: %v", path, err)
	}
	defer file.Close() //nolint:errcheck // The complete read reports failures.
	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		t.Fatalf("hash native artifact %q: %v", path, err)
	}
	if got := hex.EncodeToString(hash.Sum(nil)); got != want {
		t.Fatalf("native artifact %q SHA-256 = %s, want %s", path, got, want)
	}
}

func tensorNames(descriptors []bundle.TensorDescriptor) []string {
	names := make([]string, len(descriptors))
	for index, descriptor := range descriptors {
		names[index] = descriptor.Name
	}
	return names
}

func makeInputs(t *testing.T, tensors parity.Tensors) ([]ort.Value, func()) {
	t.Helper()
	batch := len(tensors.InputIDs)
	sequence := rectangularWidth(t, "input_ids", tensors.InputIDs)
	if got := rectangularWidth(t, "attention_mask", tensors.AttentionMask); got != sequence {
		t.Fatalf("attention_mask width = %d, want %d", got, sequence)
	}
	markers := rectangularWidth(t, "marker_pos", tensors.MarkerPos)
	if got := rectangularWidth(t, "marker_mask", tensors.MarkerMask); got != markers {
		t.Fatalf("marker_mask width = %d, want %d", got, markers)
	}
	if len(tensors.QType) != batch {
		t.Fatalf("qtype batch = %d, want %d", len(tensors.QType), batch)
	}

	inputIDs := newTensor(t, ort.NewShape(int64(batch), int64(sequence)), flatten(tensors.InputIDs))
	attentionMask := newTensor(t, ort.NewShape(int64(batch), int64(sequence)), flatten(tensors.AttentionMask))
	markerPos := newTensor(t, ort.NewShape(int64(batch), int64(markers)), flatten(tensors.MarkerPos))
	markerMask := newTensor(t, ort.NewShape(int64(batch), int64(markers)), flatten(tensors.MarkerMask))
	qtype := newTensor(t, ort.NewShape(int64(batch)), tensors.QType)
	values := []ort.Value{inputIDs, attentionMask, markerPos, markerMask, qtype}
	return values, func() { destroyValues(t, values) }
}

func newTensor[T ort.TensorData](t *testing.T, shape ort.Shape, data []T) *ort.Tensor[T] {
	t.Helper()
	tensor, err := ort.NewTensor(shape, data)
	if err != nil {
		t.Fatalf("create tensor %s: %v", shape, err)
	}
	return tensor
}

func rectangularWidth[T any](t *testing.T, name string, rows [][]T) int {
	t.Helper()
	if len(rows) == 0 || len(rows[0]) == 0 {
		t.Fatalf("%s is empty", name)
	}
	width := len(rows[0])
	for index, row := range rows {
		if len(row) != width {
			t.Fatalf("%s row %d width = %d, want %d", name, index, len(row), width)
		}
	}
	return width
}

func flatten[T any](rows [][]T) []T {
	if len(rows) == 0 {
		return nil
	}
	values := make([]T, 0, len(rows)*len(rows[0]))
	for _, row := range rows {
		values = append(values, row...)
	}
	return values
}

func destroyValues(t *testing.T, values []ort.Value) {
	t.Helper()
	for index := len(values) - 1; index >= 0; index-- {
		if values[index] == nil {
			continue
		}
		if err := values[index].Destroy(); err != nil {
			t.Errorf("destroy tensor %d: %v", index, err)
		}
		values[index] = nil
	}
}

func verifyOutputs(t *testing.T, outputs []ort.Value, run parity.Run) {
	t.Helper()
	if len(outputs) != 2 {
		t.Fatalf("output count = %d, want 2", len(outputs))
	}
	logits, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		t.Fatalf("logits type = %T, want *Tensor[float32]", outputs[0])
	}
	actLogits, ok := outputs[1].(*ort.Tensor[float32])
	if !ok {
		t.Fatalf("act_logits type = %T, want *Tensor[float32]", outputs[1])
	}
	assertShape(t, "logits", logits.GetShape(), len(run.ONNXOutputs.Logits), len(run.ONNXOutputs.Logits[0]))
	assertShape(t, "act_logits", actLogits.GetShape(), len(run.ONNXOutputs.ActLogits), len(run.ONNXOutputs.ActLogits[0]))
	assertClose(t, "logits", logits.GetData(), flatten(run.ONNXOutputs.Logits), run)
	assertClose(t, "act_logits", actLogits.GetData(), flatten(run.ONNXOutputs.ActLogits), run)
}

func assertShape(t *testing.T, name string, got ort.Shape, rows, columns int) {
	t.Helper()
	want := ort.NewShape(int64(rows), int64(columns))
	if !got.Equals(want) {
		t.Fatalf("%s shape = %s, want %s", name, got, want)
	}
}

func assertClose(t *testing.T, name string, got []float32, want []float64, run parity.Run) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s values = %d, want %d", name, len(got), len(want))
	}
	for index := range got {
		actual := float64(got[index])
		if math.IsNaN(actual) || math.IsInf(actual, 0) {
			t.Fatalf("%s[%d] is not finite: %v", name, index, actual)
		}
		delta := math.Abs(actual - want[index])
		limit := 1e-4 + 1e-4*math.Abs(want[index])
		if delta > limit {
			t.Fatalf("%s[%d] = %.9g, want %.9g (delta %.9g > %.9g) in run %s", name, index, actual, want[index], delta, limit, run.ID)
		}
	}
}
