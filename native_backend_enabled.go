//go:build laya_native && cgo

package laya

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"

	ort "github.com/yalue/onnxruntime_go"

	nativetokenizer "github.com/metalagman/laya-go/internal/native/tokenizer"
	internalsource "github.com/metalagman/laya-go/internal/source"
)

const (
	nativeLibraryEnvironment   = "LAYA_ONNXRUNTIME_LIBRARY"
	nativeTelemetryEnvironment = "ORT_DISABLE_TELEMETRY"
	nativeLibrarySize          = 28497752
	nativeLibrarySHA256        = "5715f06d8992ca8eeeddcce43df3a7d38f97d537052126f558e912cb312460ca"
)

var (
	processNativeCoordinator nativeCoordinator
	nativeRuntimeSequence    atomic.Uint64
)

func newNativeRuntimeEngine(ctx context.Context, _ RuntimeOptions) (runtimeEngine, error) {
	key, err := resolveNativeEnvironmentKey(ctx)
	if err != nil {
		return nil, err
	}
	lease, err := processNativeCoordinator.acquire(ctx, key, onnxRuntimeEnvironment{})
	if err != nil {
		return nil, err
	}
	return &nativeRuntimeEngine{
		lease:     lease,
		factory:   productionNativeResourceFactory{},
		runtimeID: RuntimeID(fmt.Sprintf("laya-native-%d", nativeRuntimeSequence.Add(1))),
	}, nil
}

func resolveNativeEnvironmentKey(ctx context.Context) (nativeEnvironmentKey, error) {
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" {
		return nativeEnvironmentKey{}, fmt.Errorf("%w: native candidate requires linux/amd64", ErrNativeUnavailable)
	}
	if !nativeTelemetryDisabled(os.Getenv(nativeTelemetryEnvironment)) {
		return nativeEnvironmentKey{}, fmt.Errorf(
			"%w: %s must be set before native initialization",
			ErrNativeUnavailable,
			nativeTelemetryEnvironment,
		)
	}
	path := os.Getenv(nativeLibraryEnvironment)
	if path == "" {
		return nativeEnvironmentKey{}, fmt.Errorf("%w: %s is not set", ErrNativeUnavailable, nativeLibraryEnvironment)
	}
	if !filepath.IsAbs(path) {
		return nativeEnvironmentKey{}, fmt.Errorf("%w: %s must be an absolute local path", ErrNativeUnavailable, nativeLibraryEnvironment)
	}
	path = filepath.Clean(path)
	file, err := os.Open(path)
	if err != nil {
		return nativeEnvironmentKey{}, newInferenceStageError(ErrNativeUnavailable, "open ONNX Runtime artifact", err)
	}
	defer file.Close() //nolint:errcheck // Complete reads report artifact failures.
	info, err := file.Stat()
	if err != nil {
		return nativeEnvironmentKey{}, newInferenceStageError(ErrNativeUnavailable, "inspect ONNX Runtime artifact", err)
	}
	if !info.Mode().IsRegular() || info.Size() != nativeLibrarySize {
		return nativeEnvironmentKey{}, fmt.Errorf("%w: ONNX Runtime artifact identity does not match", ErrNativeUnavailable)
	}
	hash := sha256.New()
	buffer := make([]byte, 64<<10)
	for {
		if err := ctx.Err(); err != nil {
			return nativeEnvironmentKey{}, err
		}
		count, readErr := file.Read(buffer)
		if count > 0 {
			_, _ = hash.Write(buffer[:count])
		}
		if errors.Is(readErr, io.EOF) {
			break
		}
		if readErr != nil {
			return nativeEnvironmentKey{}, newInferenceStageError(ErrNativeUnavailable, "read ONNX Runtime artifact", readErr)
		}
		if count == 0 {
			return nativeEnvironmentKey{}, newInferenceStageError(ErrNativeUnavailable, "read ONNX Runtime artifact", io.ErrNoProgress)
		}
	}
	digest := hex.EncodeToString(hash.Sum(nil))
	if digest != nativeLibrarySHA256 {
		return nativeEnvironmentKey{}, fmt.Errorf("%w: ONNX Runtime artifact identity does not match", ErrNativeUnavailable)
	}
	return nativeEnvironmentKey{libraryPath: path, librarySHA256: digest}, nil
}

func nativeTelemetryDisabled(value string) bool {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "1", "true", "yes", "on", "y":
		return true
	default:
		return false
	}
}

type onnxRuntimeEnvironment struct{}

func (onnxRuntimeEnvironment) SetSharedLibraryPath(path string) { ort.SetSharedLibraryPath(path) }
func (onnxRuntimeEnvironment) Initialize() error {
	if err := ort.InitializeEnvironment(ort.WithLogLevelError()); err != nil {
		return err
	}
	if err := ort.DisableTelemetry(); err != nil {
		return errors.Join(err, ort.DestroyEnvironment())
	}
	return nil
}
func (onnxRuntimeEnvironment) Destroy() error {
	err := ort.DestroyEnvironment()
	if errors.Is(err, ort.NotInitializedError) {
		return nil
	}
	return err
}
func (onnxRuntimeEnvironment) Version() string { return ort.GetVersion() }

type productionNativeResourceFactory struct{}

func (productionNativeResourceFactory) OpenDirectorySource(
	ctx context.Context,
	dir string,
) (nativeBundleSource, error) {
	if err := validateContext(ctx); err != nil {
		return nil, err
	}
	return internalsource.OpenDirectory(ctx, dir)
}

func (productionNativeResourceFactory) OpenFilesystemSource(
	ctx context.Context,
	fileSystem fs.FS,
	opts FSModelOptions,
) (nativeBundleSource, error) {
	if err := validateContext(ctx); err != nil {
		return nil, err
	}
	return internalsource.Materialize(ctx, fileSystem, opts.Root, opts.WorkDir)
}

func (productionNativeResourceFactory) OpenTokenizer(path string) (nativeTokenizerResource, error) {
	tokenizer, err := nativetokenizer.FromFile(path)
	if err != nil {
		return nil, err
	}
	return &localTokenizerResource{tokenizer: tokenizer}, nil
}

func (productionNativeResourceFactory) OpenSession(
	path string,
	inputNames []string,
	outputNames []string,
) (nativeSessionResource, error) {
	session, err := ort.NewDynamicAdvancedSession(path, inputNames, outputNames, nil)
	if err != nil {
		return nil, err
	}
	return &localSessionResource{session: session}, nil
}

type localTokenizerResource struct {
	tokenizer *nativetokenizer.Tokenizer
}

func (r *localTokenizerResource) Encode(text string) ([]uint32, error) {
	return r.tokenizer.EncodeErr(text, false)
}

func (r *localTokenizerResource) Close() error {
	return r.tokenizer.Close()
}

type localSessionResource struct {
	session *ort.DynamicAdvancedSession
}

func (r *localSessionResource) Run(ctx context.Context, batch preparedBatch) (rawModelOutputs, error) {
	rows, sequenceWidth, markerWidth, err := preparedTensorDimensions(batch)
	if err != nil {
		return rawModelOutputs{}, err
	}
	if err := ctx.Err(); err != nil {
		return rawModelOutputs{}, err
	}
	inputs, err := newORTInputs(batch, rows, sequenceWidth, markerWidth)
	if err != nil {
		return rawModelOutputs{}, err
	}
	runOptions, err := ort.NewRunOptions()
	if err != nil {
		return rawModelOutputs{}, newInferenceStageError(
			ErrNativeFailure,
			"allocate ONNX run options",
			errors.Join(err, destroyORTValues(inputs)),
		)
	}
	return executeCancelableNativeCall(ctx, &localNativeCall{
		session:     r.session,
		inputs:      inputs,
		outputs:     make([]ort.Value, 2),
		runOptions:  runOptions,
		rows:        rows,
		markerWidth: markerWidth,
	})
}

func (r *localSessionResource) Close() error {
	return r.session.Destroy()
}

type localNativeCall struct {
	session     *ort.DynamicAdvancedSession
	inputs      []ort.Value
	outputs     []ort.Value
	runOptions  *ort.RunOptions
	rows        int
	markerWidth int
}

func (c *localNativeCall) Execute() (rawModelOutputs, error) {
	if err := c.session.RunWithOptions(c.inputs, c.outputs, c.runOptions); err != nil {
		return rawModelOutputs{}, err
	}
	return copyORTOutputs(c.outputs, c.rows, c.markerWidth)
}

func (c *localNativeCall) Terminate() error {
	return c.runOptions.Terminate()
}

func (c *localNativeCall) Close() error {
	outputErr := destroyORTValues(c.outputs)
	optionsErr := c.runOptions.Destroy()
	c.runOptions = nil
	inputErr := destroyORTValues(c.inputs)
	return errors.Join(outputErr, optionsErr, inputErr)
}

func preparedTensorDimensions(batch preparedBatch) (rows, sequenceWidth, markerWidth int, err error) {
	rows = len(batch.items)
	if rows == 0 || len(batch.inputIDs) != rows || len(batch.attentionMask) != rows ||
		len(batch.markerPos) != rows || len(batch.markerMask) != rows || len(batch.qtype) != rows {
		return 0, 0, 0, fmt.Errorf("%w: prepared tensor batch dimensions are inconsistent", ErrNativeFailure)
	}
	sequenceWidth = len(batch.inputIDs[0])
	markerWidth = len(batch.markerPos[0])
	if sequenceWidth == 0 || markerWidth == 0 {
		return 0, 0, 0, fmt.Errorf("%w: prepared tensor widths are empty", ErrNativeFailure)
	}
	for row := range rows {
		if len(batch.inputIDs[row]) != sequenceWidth || len(batch.attentionMask[row]) != sequenceWidth ||
			len(batch.markerPos[row]) != markerWidth || len(batch.markerMask[row]) != markerWidth {
			return 0, 0, 0, fmt.Errorf("%w: prepared tensor rows are not rectangular", ErrNativeFailure)
		}
	}
	return rows, sequenceWidth, markerWidth, nil
}

func newORTInputs(
	batch preparedBatch,
	rows int,
	sequenceWidth int,
	markerWidth int,
) ([]ort.Value, error) {
	values := make([]ort.Value, 0, 5)
	add := func(value ort.Value, err error) error {
		if err != nil {
			return err
		}
		values = append(values, value)
		return nil
	}
	sequenceShape := ort.NewShape(int64(rows), int64(sequenceWidth))
	markerShape := ort.NewShape(int64(rows), int64(markerWidth))
	if err := add(ort.NewTensor(sequenceShape, flattenTensorRows(batch.inputIDs))); err != nil {
		return nil, cleanupFailedORTInputs(values, err)
	}
	if err := add(ort.NewTensor(sequenceShape, flattenTensorRows(batch.attentionMask))); err != nil {
		return nil, cleanupFailedORTInputs(values, err)
	}
	if err := add(ort.NewTensor(markerShape, flattenTensorRows(batch.markerPos))); err != nil {
		return nil, cleanupFailedORTInputs(values, err)
	}
	if err := add(ort.NewTensor(markerShape, flattenTensorRows(batch.markerMask))); err != nil {
		return nil, cleanupFailedORTInputs(values, err)
	}
	if err := add(ort.NewTensor(ort.NewShape(int64(rows)), append([]int64(nil), batch.qtype...))); err != nil {
		return nil, cleanupFailedORTInputs(values, err)
	}
	return values, nil
}

func cleanupFailedORTInputs(values []ort.Value, cause error) error {
	return newInferenceStageError(ErrNativeFailure, "allocate ONNX input tensors", errors.Join(cause, destroyORTValues(values)))
}

func copyORTOutputs(values []ort.Value, rows int, markerWidth int) (rawModelOutputs, error) {
	if len(values) != 2 {
		return rawModelOutputs{}, fmt.Errorf("%w: ONNX output count is %d, want 2", ErrInvalidOutput, len(values))
	}
	logits, ok := values[0].(*ort.Tensor[float32])
	if !ok {
		return rawModelOutputs{}, fmt.Errorf("%w: logits output has an incompatible tensor type", ErrInvalidOutput)
	}
	actionLogits, ok := values[1].(*ort.Tensor[float32])
	if !ok {
		return rawModelOutputs{}, fmt.Errorf("%w: action output has an incompatible tensor type", ErrInvalidOutput)
	}
	if !logits.GetShape().Equals(ort.NewShape(int64(rows), int64(markerWidth))) {
		return rawModelOutputs{}, fmt.Errorf("%w: logits output shape does not match the prepared batch", ErrInvalidOutput)
	}
	if !actionLogits.GetShape().Equals(ort.NewShape(int64(rows), 2)) {
		return rawModelOutputs{}, fmt.Errorf("%w: action output shape does not match the prepared batch", ErrInvalidOutput)
	}
	return rawModelOutputs{
		logits:       splitFloat32Rows(logits.GetData(), rows, markerWidth),
		actionLogits: splitFloat32Rows(actionLogits.GetData(), rows, 2),
	}, nil
}

func destroyORTValues(values []ort.Value) error {
	var cleanupErr error
	for index := len(values) - 1; index >= 0; index-- {
		if values[index] == nil {
			continue
		}
		cleanupErr = errors.Join(cleanupErr, values[index].Destroy())
		values[index] = nil
	}
	return cleanupErr
}

func flattenTensorRows[T any](rows [][]T) []T {
	if len(rows) == 0 {
		return nil
	}
	values := make([]T, 0, len(rows)*len(rows[0]))
	for _, row := range rows {
		values = append(values, row...)
	}
	return values
}

func splitFloat32Rows(values []float32, rows int, width int) [][]float32 {
	result := make([][]float32, rows)
	for row := range rows {
		start := row * width
		result[row] = append([]float32(nil), values[start:start+width]...)
	}
	return result
}
