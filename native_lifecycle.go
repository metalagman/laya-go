package laya

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"sync"

	"github.com/metalagman/laya-go/internal/bundle"
	internalsource "github.com/metalagman/laya-go/internal/source"
)

const nativeONNXRuntimeVersion = "1.29.0"

type nativeEnvironment interface {
	SetSharedLibraryPath(string)
	Initialize() error
	Destroy() error
	Version() string
}

type nativeEnvironmentKey struct {
	libraryPath   string
	librarySHA256 string
}

type nativeCoordinator struct {
	mu sync.Mutex

	environment    nativeEnvironment
	key            nativeEnvironmentKey
	leases         int
	cleanupPending bool
}

type nativeLease struct {
	mu sync.Mutex

	coordinator *nativeCoordinator
	released    bool
}

func (c *nativeCoordinator) acquire(
	ctx context.Context,
	key nativeEnvironmentKey,
	environment nativeEnvironment,
) (*nativeLease, error) {
	if err := validateContext(ctx); err != nil {
		return nil, err
	}
	if environment == nil || key.libraryPath == "" || key.librarySHA256 == "" {
		return nil, fmt.Errorf("%w: native environment configuration is incomplete", ErrInvalidConfig)
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if c.cleanupPending {
		if c.leases > 0 {
			return nil, fmt.Errorf("%w: ONNX Runtime cleanup must be retried by its owner", ErrNativeFailure)
		}
		if err := c.environment.Destroy(); err != nil {
			return nil, newInferenceStageError(ErrNativeFailure, "retry ONNX Runtime cleanup", err)
		}
		c.environment = nil
		c.key = nativeEnvironmentKey{}
		c.cleanupPending = false
	}
	if c.leases > 0 {
		if c.key != key {
			return nil, fmt.Errorf("%w: ONNX Runtime is already initialized with different local artifact", ErrInvalidConfig)
		}
		c.leases++
		return &nativeLease{coordinator: c}, nil
	}

	environment.SetSharedLibraryPath(key.libraryPath)
	if err := environment.Initialize(); err != nil {
		return nil, newInferenceStageError(ErrNativeUnavailable, "initialize ONNX Runtime", err)
	}
	if version := environment.Version(); version != nativeONNXRuntimeVersion {
		cleanupErr := environment.Destroy()
		cause := fmt.Errorf("runtime version mismatch")
		if cleanupErr != nil {
			cause = errors.Join(cause, cleanupErr)
			c.environment = environment
			c.key = key
			c.cleanupPending = true
		}
		return nil, newInferenceStageError(ErrNativeUnavailable, "validate ONNX Runtime version", cause)
	}
	c.environment = environment
	c.key = key
	c.leases = 1
	return &nativeLease{coordinator: c}, nil
}

func (l *nativeLease) close(ctx context.Context) error {
	if err := validateContext(ctx); err != nil {
		return err
	}
	if l == nil {
		return fmt.Errorf("%w: native lease is nil", ErrInvalidState)
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.released {
		return nil
	}
	coordinator := l.coordinator
	if coordinator == nil {
		return fmt.Errorf("%w: native lease has no coordinator", ErrInvalidState)
	}
	coordinator.mu.Lock()
	defer coordinator.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return err
	}
	if coordinator.leases <= 0 {
		return fmt.Errorf("%w: native coordinator has no active leases", ErrInvalidState)
	}
	if coordinator.leases > 1 {
		coordinator.leases--
		l.released = true
		return nil
	}
	if err := coordinator.environment.Destroy(); err != nil {
		coordinator.cleanupPending = true
		return newInferenceStageError(ErrNativeFailure, "destroy ONNX Runtime", err)
	}
	coordinator.environment = nil
	coordinator.key = nativeEnvironmentKey{}
	coordinator.leases = 0
	coordinator.cleanupPending = false
	l.released = true
	return nil
}

type nativeBundleSource interface {
	Manifest() bundle.Manifest
	Path(string) (string, error)
	ReadFile(context.Context, string, int64) ([]byte, error)
	Close() error
}

type nativeTokenizerResource interface {
	Encode(string) ([]uint32, error)
	Close() error
}

type nativeSessionResource interface {
	Run(context.Context, preparedBatch) (rawModelOutputs, error)
	Close() error
}

type nativeResourceFactory interface {
	OpenDirectorySource(context.Context, string) (nativeBundleSource, error)
	OpenFilesystemSource(context.Context, fs.FS, FSModelOptions) (nativeBundleSource, error)
	OpenTokenizer(string) (nativeTokenizerResource, error)
	OpenSession(string, []string, []string) (nativeSessionResource, error)
}

type nativeRuntimeEngine struct {
	mu sync.Mutex

	lease     *nativeLease
	factory   nativeResourceFactory
	runtimeID RuntimeID
	closed    bool
}

func (e *nativeRuntimeEngine) OpenModelDir(
	ctx context.Context,
	dir string,
	opts ModelOptions,
) (modelEngine, error) {
	if err := validateContext(ctx); err != nil {
		return nil, err
	}
	if e == nil || e.factory == nil {
		return nil, fmt.Errorf("%w: native runtime is not initialized", ErrInvalidState)
	}
	e.mu.Lock()
	closed := e.closed
	e.mu.Unlock()
	if closed {
		return nil, ErrInvalidState
	}

	source, err := e.factory.OpenDirectorySource(ctx, dir)
	if err != nil {
		return nil, nativeSourceError(ctx, "open directory source", err)
	}
	return e.openVerified(ctx, source, opts)
}

func (e *nativeRuntimeEngine) OpenModelFS(ctx context.Context, fileSystem fs.FS, opts FSModelOptions) (modelEngine, error) {
	if err := validateContext(ctx); err != nil {
		return nil, err
	}
	if e == nil || e.factory == nil {
		return nil, fmt.Errorf("%w: native runtime is not initialized", ErrInvalidState)
	}
	e.mu.Lock()
	closed := e.closed
	e.mu.Unlock()
	if closed {
		return nil, ErrInvalidState
	}
	source, err := e.factory.OpenFilesystemSource(ctx, fileSystem, opts)
	if err != nil {
		return nil, nativeSourceError(ctx, "open filesystem source", err)
	}
	return e.openVerified(ctx, source, opts.ModelOptions)
}

func (e *nativeRuntimeEngine) openVerified(
	ctx context.Context,
	source nativeBundleSource,
	opts ModelOptions,
) (modelEngine, error) {
	if source == nil {
		return nil, fmt.Errorf("%w: verified bundle source is nil", ErrInvalidState)
	}
	manifest := source.Manifest()
	calibrationBytes, err := source.ReadFile(ctx, manifest.Calibration.ConfigPath, maxCalibrationConfigBytes)
	if err != nil {
		return nil, nativeOpenFailure(sourceErrorCategory(err), "read calibration config", err, nil, nil, source)
	}
	calibration, err := parseCalibrationConfig(manifest, calibrationBytes)
	if err != nil {
		return nil, nativeOpenFailure(errorCategory(err, ErrInvalidBundle), "parse calibration config", err, nil, nil, source)
	}
	if err := ctx.Err(); err != nil {
		return nil, nativeOpenFailure(err, "cancel model open", err, nil, nil, source)
	}
	tokenizerPath, err := source.Path(manifest.Tokenizer.JSONPath)
	if err != nil {
		return nil, nativeOpenFailure(sourceErrorCategory(err), "resolve tokenizer path", err, nil, nil, source)
	}
	tokenizer, err := e.factory.OpenTokenizer(tokenizerPath)
	if err != nil {
		return nil, nativeOpenFailure(ErrNativeFailure, "open local tokenizer", err, nil, nil, source)
	}
	if err := ctx.Err(); err != nil {
		return nil, nativeOpenFailure(err, "cancel model open", err, nil, tokenizer, source)
	}
	modelPath, err := source.Path(manifest.Model.Path)
	if err != nil {
		return nil, nativeOpenFailure(sourceErrorCategory(err), "resolve model path", err, nil, tokenizer, source)
	}
	session, err := e.factory.OpenSession(
		modelPath,
		tensorNamesFromManifest(manifest.Model.Inputs),
		tensorNamesFromManifest(manifest.Model.Outputs),
	)
	if err != nil {
		return nil, nativeOpenFailure(ErrNativeFailure, "open ONNX session", err, nil, tokenizer, source)
	}
	if err := ctx.Err(); err != nil {
		return nil, nativeOpenFailure(err, "cancel model open", err, session, tokenizer, source)
	}

	return &nativeModelEngine{
		source:      source,
		tokenizer:   tokenizer,
		session:     session,
		manifest:    manifest,
		calibration: calibration,
		modelOpts:   opts,
		runtimeID:   e.runtimeID,
	}, nil
}

func (e *nativeRuntimeEngine) Close(ctx context.Context) error {
	if err := validateContext(ctx); err != nil {
		return err
	}
	if e == nil {
		return fmt.Errorf("%w: native runtime is nil", ErrInvalidState)
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil
	}
	if e.lease == nil {
		return fmt.Errorf("%w: native runtime has no environment lease", ErrInvalidState)
	}
	if err := e.lease.close(ctx); err != nil {
		return err
	}
	e.closed = true
	e.lease = nil
	return nil
}

type nativeModelEngine struct {
	mu sync.Mutex

	source      nativeBundleSource
	tokenizer   nativeTokenizerResource
	session     nativeSessionResource
	manifest    bundle.Manifest
	calibration calibrationConfig
	modelOpts   ModelOptions
	runtimeID   RuntimeID
	closed      bool
}

func (e *nativeModelEngine) Predict(
	ctx context.Context,
	state State,
	questions []Question,
) (Prediction, error) {
	if err := validateContext(ctx); err != nil {
		return Prediction{}, err
	}
	if e == nil {
		return Prediction{}, fmt.Errorf("%w: native model is nil", ErrInvalidState)
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return Prediction{}, ErrModelClosed
	}
	if e.tokenizer == nil || e.session == nil {
		return Prediction{}, fmt.Errorf("%w: native model resources are incomplete", ErrInvalidState)
	}
	batch, err := prepareInferenceBatch(state, questions, e.manifest, e.tokenizer, e.modelOpts)
	if err != nil {
		return Prediction{}, err
	}
	if err := ctx.Err(); err != nil {
		return Prediction{}, err
	}
	outputs, err := e.session.Run(ctx, batch)
	if err != nil {
		if contextErr := ctx.Err(); contextErr != nil {
			return Prediction{}, newInferenceStageError(
				contextErr,
				"cancel inference",
				errors.Join(contextErr, err),
			)
		}
		if errors.Is(err, ErrInvalidOutput) {
			return Prediction{}, newInferenceStageError(ErrInvalidOutput, "validate ONNX outputs", err)
		}
		return Prediction{}, newInferenceStageError(ErrNativeFailure, "run ONNX session", err)
	}
	if err := ctx.Err(); err != nil {
		return Prediction{}, err
	}
	return formatInferencePrediction(batch, outputs, e.calibration, PredictionMetadata{
		ModelID:   ModelID(e.manifest.Bundle.ID),
		BundleID:  BundleID(e.manifest.ID()),
		RuntimeID: e.runtimeID,
	})
}

func (e *nativeModelEngine) Close(ctx context.Context) error {
	if err := validateContext(ctx); err != nil {
		return err
	}
	if e == nil {
		return fmt.Errorf("%w: native model is nil", ErrInvalidState)
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil
	}
	if e.session != nil {
		if err := e.session.Close(); err != nil {
			return newInferenceStageError(ErrNativeFailure, "close ONNX session", err)
		}
		e.session = nil
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if e.tokenizer != nil {
		if err := e.tokenizer.Close(); err != nil {
			return newInferenceStageError(ErrNativeFailure, "close tokenizer", err)
		}
		e.tokenizer = nil
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if e.source != nil {
		if err := e.source.Close(); err != nil {
			return newInferenceStageError(ErrMaterialization, "close model source", err)
		}
		e.source = nil
	}
	e.closed = true
	return nil
}

func tensorNamesFromManifest(descriptors []bundle.TensorDescriptor) []string {
	names := make([]string, len(descriptors))
	for index, descriptor := range descriptors {
		names[index] = descriptor.Name
	}
	return names
}

func nativeSourceError(ctx context.Context, stage string, err error) error {
	if contextErr := ctx.Err(); contextErr != nil {
		return newInferenceStageError(contextErr, stage, errors.Join(contextErr, err))
	}
	return newInferenceStageError(sourceErrorCategory(err), stage, err)
}

func sourceErrorCategory(err error) error {
	switch {
	case errors.Is(err, ErrUnsupportedSource):
		return ErrUnsupportedSource
	case errors.Is(err, ErrUnsupportedBundle):
		return ErrUnsupportedBundle
	case errors.Is(err, ErrIntegrity):
		return ErrIntegrity
	case errors.Is(err, ErrMaterialization):
		return ErrMaterialization
	case errors.Is(err, internalsource.ErrClosed):
		return ErrInvalidState
	default:
		return ErrInvalidBundle
	}
}

func errorCategory(err, fallback error) error {
	for _, category := range []error{
		context.Canceled,
		context.DeadlineExceeded,
		ErrUnsupportedSource,
		ErrUnsupportedBundle,
		ErrIntegrity,
		ErrMaterialization,
		ErrInvalidBundle,
		ErrInvalidConfig,
	} {
		if errors.Is(err, category) {
			return category
		}
	}
	return fallback
}

func nativeOpenFailure(
	category error,
	stage string,
	cause error,
	session nativeSessionResource,
	tokenizer nativeTokenizerResource,
	source nativeBundleSource,
) error {
	errs := []error{cause}
	if session != nil {
		errs = append(errs, session.Close())
	}
	if tokenizer != nil {
		errs = append(errs, tokenizer.Close())
	}
	if source != nil {
		errs = append(errs, source.Close())
	}
	return newInferenceStageError(category, stage, errors.Join(errs...))
}
