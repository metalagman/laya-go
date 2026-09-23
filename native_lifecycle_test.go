package laya

import (
	"context"
	"errors"
	"io/fs"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/metalagman/laya-go/internal/bundle"
)

func TestNativeCoordinatorSharesCompatibleEnvironment(t *testing.T) {
	coordinator := &nativeCoordinator{}
	key := nativeEnvironmentKey{libraryPath: "/runtime/libonnxruntime.so", librarySHA256: "digest"}
	primary := &fakeNativeEnvironment{version: nativeONNXRuntimeVersion}
	unused := &fakeNativeEnvironment{version: nativeONNXRuntimeVersion}

	first, err := coordinator.acquire(t.Context(), key, primary)
	if err != nil {
		t.Fatalf("first acquire returned unexpected error: %v", err)
	}
	second, err := coordinator.acquire(t.Context(), key, unused)
	if err != nil {
		t.Fatalf("second acquire returned unexpected error: %v", err)
	}
	if got := primary.snapshot(); got != (nativeEnvironmentSnapshot{path: key.libraryPath, initializeCalls: 1}) {
		t.Errorf("primary environment after acquire = %+v", got)
	}
	if got := unused.snapshot(); got != (nativeEnvironmentSnapshot{}) {
		t.Errorf("unused environment after acquire = %+v, want zero", got)
	}

	conflicting := nativeEnvironmentKey{libraryPath: "/other/libonnxruntime.so", librarySHA256: "other"}
	if _, err := coordinator.acquire(t.Context(), conflicting, unused); !errors.Is(err, ErrInvalidConfig) {
		t.Errorf("conflicting acquire error = %v, want errors.Is(_, ErrInvalidConfig)", err)
	}
	if err := first.close(t.Context()); err != nil {
		t.Fatalf("first lease close returned unexpected error: %v", err)
	}
	if got := primary.snapshot().destroyCalls; got != 0 {
		t.Errorf("destroy calls after non-final close = %d, want 0", got)
	}
	if err := second.close(t.Context()); err != nil {
		t.Fatalf("final lease close returned unexpected error: %v", err)
	}
	if err := second.close(t.Context()); err != nil {
		t.Fatalf("repeated lease close returned unexpected error: %v", err)
	}
	if got := primary.snapshot().destroyCalls; got != 1 {
		t.Errorf("destroy calls = %d, want 1", got)
	}
}

func TestNativeCoordinatorFailureRecovery(t *testing.T) {
	key := nativeEnvironmentKey{libraryPath: "/runtime/libonnxruntime.so", librarySHA256: "digest"}
	tests := []struct {
		name        string
		environment *fakeNativeEnvironment
		wantError   error
		wantDestroy int
	}{
		{
			name:        "initialize",
			environment: &fakeNativeEnvironment{version: nativeONNXRuntimeVersion, initializeErr: errors.New("secret initialize detail")},
			wantError:   ErrNativeUnavailable,
		},
		{
			name:        "version",
			environment: &fakeNativeEnvironment{version: "1.28.0"},
			wantError:   ErrNativeUnavailable,
			wantDestroy: 1,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			coordinator := &nativeCoordinator{}
			lease, err := coordinator.acquire(t.Context(), key, test.environment)
			if lease != nil {
				t.Errorf("acquire lease = %v, want nil", lease)
			}
			if !errors.Is(err, test.wantError) {
				t.Errorf("acquire error = %v, want errors.Is(_, %v)", err, test.wantError)
			}
			if strings.Contains(err.Error(), "secret") {
				t.Errorf("acquire error disclosed underlying detail: %v", err)
			}
			if got := test.environment.snapshot().destroyCalls; got != test.wantDestroy {
				t.Errorf("destroy calls = %d, want %d", got, test.wantDestroy)
			}

			recovery := &fakeNativeEnvironment{version: nativeONNXRuntimeVersion}
			recovered, recoveryErr := coordinator.acquire(t.Context(), key, recovery)
			if recoveryErr != nil {
				t.Fatalf("recovery acquire returned unexpected error: %v", recoveryErr)
			}
			if err := recovered.close(t.Context()); err != nil {
				t.Fatalf("recovery close returned unexpected error: %v", err)
			}
		})
	}
}

func TestNativeCoordinatorRetriesFailedFinalDestroy(t *testing.T) {
	coordinator := &nativeCoordinator{}
	key := nativeEnvironmentKey{libraryPath: "/runtime/libonnxruntime.so", librarySHA256: "digest"}
	environment := &fakeNativeEnvironment{
		version:     nativeONNXRuntimeVersion,
		destroyErrs: []error{errors.New("secret destroy detail"), nil},
	}
	lease, err := coordinator.acquire(t.Context(), key, environment)
	if err != nil {
		t.Fatalf("acquire returned unexpected error: %v", err)
	}
	if err := lease.close(t.Context()); !errors.Is(err, ErrNativeFailure) {
		t.Fatalf("first close error = %v, want errors.Is(_, ErrNativeFailure)", err)
	} else if strings.Contains(err.Error(), "secret") {
		t.Errorf("close error disclosed underlying detail: %v", err)
	}
	unused := &fakeNativeEnvironment{version: nativeONNXRuntimeVersion}
	if acquired, err := coordinator.acquire(t.Context(), key, unused); acquired != nil || !errors.Is(err, ErrNativeFailure) {
		t.Errorf("acquire during pending cleanup = (%v, %v), want nil and ErrNativeFailure", acquired, err)
	}
	if got := unused.snapshot().initializeCalls; got != 0 {
		t.Errorf("initialize calls during pending cleanup = %d, want 0", got)
	}
	if err := lease.close(t.Context()); err != nil {
		t.Fatalf("retry close returned unexpected error: %v", err)
	}
	if got := environment.snapshot().destroyCalls; got != 2 {
		t.Errorf("destroy calls = %d, want 2", got)
	}
}

func TestNativeCoordinatorConcurrentLeaseRelease(t *testing.T) {
	coordinator := &nativeCoordinator{}
	key := nativeEnvironmentKey{libraryPath: "/runtime/libonnxruntime.so", librarySHA256: "digest"}
	environment := &fakeNativeEnvironment{version: nativeONNXRuntimeVersion}
	const leaseCount = 32
	leases := make([]*nativeLease, leaseCount)
	for index := range leases {
		lease, err := coordinator.acquire(t.Context(), key, environment)
		if err != nil {
			t.Fatalf("acquire %d returned unexpected error: %v", index, err)
		}
		leases[index] = lease
	}

	start := make(chan struct{})
	errorsByLease := make(chan error, leaseCount)
	var group sync.WaitGroup
	for _, lease := range leases {
		group.Add(1)
		go func(lease *nativeLease) {
			defer group.Done()
			<-start
			errorsByLease <- lease.close(context.Background())
		}(lease)
	}
	close(start)
	group.Wait()
	close(errorsByLease)
	for err := range errorsByLease {
		if err != nil {
			t.Errorf("concurrent close returned unexpected error: %v", err)
		}
	}
	if got := environment.snapshot(); got.initializeCalls != 1 || got.destroyCalls != 1 {
		t.Errorf("environment calls after concurrent release = %+v, want one initialize and destroy", got)
	}
}

func TestNativeRuntimeOpenModelLifecycle(t *testing.T) {
	manifest := loadTransformManifest(t)
	factory := newFakeNativeFactory(manifest)
	engine := &nativeRuntimeEngine{factory: factory, runtimeID: "runtime-test"}
	opts := ModelOptions{InputTokenLimit: 128, Truncation: TruncateOverflow}

	firstEngine, err := engine.OpenModelDir(t.Context(), "/caller/bundle", opts)
	if err != nil {
		t.Fatalf("first OpenModelDir returned unexpected error: %v", err)
	}
	secondEngine, err := engine.OpenModelDir(t.Context(), "/caller/bundle", opts)
	if err != nil {
		t.Fatalf("second OpenModelDir returned unexpected error: %v", err)
	}
	first := firstEngine.(*nativeModelEngine)
	second := secondEngine.(*nativeModelEngine)
	if first.tokenizer == second.tokenizer || first.session == second.session {
		t.Fatal("independent model opens shared tokenizer or session resources")
	}
	if first.runtimeID != "runtime-test" || first.modelOpts != opts {
		t.Errorf("model metadata = runtime %q opts %+v", first.runtimeID, first.modelOpts)
	}
	if err := first.Close(t.Context()); err != nil {
		t.Fatalf("first model close returned unexpected error: %v", err)
	}
	if err := first.Close(t.Context()); err != nil {
		t.Fatalf("repeated model close returned unexpected error: %v", err)
	}
	if err := second.Close(t.Context()); err != nil {
		t.Fatalf("second model close returned unexpected error: %v", err)
	}

	want := []string{
		"verify", "read:rl_agent_config.json", "tokenizer:/verified/tokenizer/tokenizer.json", "session:/verified/model.onnx",
		"verify", "read:rl_agent_config.json", "tokenizer:/verified/tokenizer/tokenizer.json", "session:/verified/model.onnx",
		"session.close", "tokenizer.close", "source.close", "session.close", "tokenizer.close", "source.close",
	}
	if got := factory.eventSnapshot(); !reflect.DeepEqual(got, want) {
		t.Errorf("events = %v, want %v", got, want)
	}
	if !reflect.DeepEqual(factory.inputNames, tensorNamesFromManifest(manifest.Model.Inputs)) {
		t.Errorf("session input names = %v", factory.inputNames)
	}
	if !reflect.DeepEqual(factory.outputNames, tensorNamesFromManifest(manifest.Model.Outputs)) {
		t.Errorf("session output names = %v", factory.outputNames)
	}
}

func TestNativeRuntimeSourceModesUseIdenticalOpenPipeline(t *testing.T) {
	type openModel func(*nativeRuntimeEngine) (modelEngine, error)
	tests := []struct {
		name       string
		open       openModel
		firstEvent string
	}{
		{
			name: "directory",
			open: func(engine *nativeRuntimeEngine) (modelEngine, error) {
				return engine.OpenModelDir(t.Context(), "/caller/bundle", ModelOptions{})
			},
			firstEvent: "verify",
		},
		{
			name: "filesystem",
			open: func(engine *nativeRuntimeEngine) (modelEngine, error) {
				return engine.OpenModelFS(t.Context(), fstestFS{}, FSModelOptions{WorkDir: t.TempDir()})
			},
			firstEvent: "materialize",
		},
	}
	wantTail := []string{
		"read:rl_agent_config.json",
		"tokenizer:/verified/tokenizer/tokenizer.json",
		"session:/verified/model.onnx",
		"session.close",
		"tokenizer.close",
		"source.close",
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			factory := newFakeNativeFactory(loadTransformManifest(t))
			engine := &nativeRuntimeEngine{factory: factory, runtimeID: "runtime-test"}
			model, err := test.open(engine)
			if err != nil {
				t.Fatalf("open returned unexpected error: %v", err)
			}
			if err := model.Close(t.Context()); err != nil {
				t.Fatalf("model.Close() returned unexpected error: %v", err)
			}
			want := append([]string{test.firstEvent}, wantTail...)
			if got := factory.eventSnapshot(); !reflect.DeepEqual(got, want) {
				t.Errorf("events = %v, want %v", got, want)
			}
		})
	}
}

func TestNativeRuntimeOpenModelFailureUnwind(t *testing.T) {
	secret := errors.New("secret native detail")
	tests := []struct {
		name       string
		configure  func(*fakeNativeFactory)
		wantError  error
		wantEvents []string
	}{
		{
			name: "verify before native",
			configure: func(factory *fakeNativeFactory) {
				factory.verifyErr = secret
			},
			wantError:  ErrInvalidBundle,
			wantEvents: []string{"verify"},
		},
		{
			name: "read before native",
			configure: func(factory *fakeNativeFactory) {
				factory.readErr = secret
			},
			wantError:  ErrInvalidBundle,
			wantEvents: []string{"verify", "read:rl_agent_config.json", "source.close"},
		},
		{
			name: "invalid calibration before native",
			configure: func(factory *fakeNativeFactory) {
				factory.config = []byte(`{"temperature":[1]}`)
			},
			wantError:  ErrUnsupportedBundle,
			wantEvents: []string{"verify", "read:rl_agent_config.json", "source.close"},
		},
		{
			name: "tokenizer",
			configure: func(factory *fakeNativeFactory) {
				factory.tokenizerErr = secret
			},
			wantError: ErrNativeFailure,
			wantEvents: []string{
				"verify", "read:rl_agent_config.json", "tokenizer:/verified/tokenizer/tokenizer.json", "source.close",
			},
		},
		{
			name: "tokenizer path",
			configure: func(factory *fakeNativeFactory) {
				factory.pathErr = secret
				factory.pathErrAt = factory.manifest.Tokenizer.JSONPath
			},
			wantError: ErrInvalidBundle,
			wantEvents: []string{
				"verify", "read:rl_agent_config.json", "source.close",
			},
		},
		{
			name: "model path unwinds tokenizer",
			configure: func(factory *fakeNativeFactory) {
				factory.pathErr = secret
				factory.pathErrAt = factory.manifest.Model.Path
			},
			wantError: ErrInvalidBundle,
			wantEvents: []string{
				"verify", "read:rl_agent_config.json", "tokenizer:/verified/tokenizer/tokenizer.json",
				"tokenizer.close", "source.close",
			},
		},
		{
			name: "session unwinds tokenizer",
			configure: func(factory *fakeNativeFactory) {
				factory.sessionErr = secret
			},
			wantError: ErrNativeFailure,
			wantEvents: []string{
				"verify", "read:rl_agent_config.json", "tokenizer:/verified/tokenizer/tokenizer.json",
				"session:/verified/model.onnx", "tokenizer.close", "source.close",
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			factory := newFakeNativeFactory(loadTransformManifest(t))
			test.configure(factory)
			engine := &nativeRuntimeEngine{factory: factory}
			model, err := engine.OpenModelDir(t.Context(), "/caller/bundle", ModelOptions{})
			if model != nil {
				t.Errorf("OpenModelDir model = %v, want nil", model)
			}
			if !errors.Is(err, test.wantError) {
				t.Errorf("OpenModelDir error = %v, want errors.Is(_, %v)", err, test.wantError)
			}
			if strings.Contains(err.Error(), "secret") {
				t.Errorf("OpenModelDir disclosed underlying detail: %v", err)
			}
			if got := factory.eventSnapshot(); !reflect.DeepEqual(got, test.wantEvents) {
				t.Errorf("events = %v, want %v", got, test.wantEvents)
			}
		})
	}
}

func TestNativeRuntimeOpenModelCancellationUnwindsResources(t *testing.T) {
	for _, stage := range []string{"source", "tokenizer", "session"} {
		t.Run(stage, func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			factory := newFakeNativeFactory(loadTransformManifest(t))
			factory.cancel = cancel
			factory.cancelStage = stage
			engine := &nativeRuntimeEngine{factory: factory}
			model, err := engine.OpenModelDir(ctx, "/caller/bundle", ModelOptions{})
			if model != nil {
				t.Errorf("OpenModelDir model = %v, want nil", model)
			}
			if !errors.Is(err, context.Canceled) {
				t.Errorf("OpenModelDir error = %v, want errors.Is(_, context.Canceled)", err)
			}
			wantTail := []string{"source.close"}
			if stage == "tokenizer" {
				wantTail = []string{"tokenizer.close", "source.close"}
			}
			if stage == "session" {
				wantTail = []string{"session.close", "tokenizer.close", "source.close"}
			}
			events := factory.eventSnapshot()
			if len(events) < len(wantTail) || !reflect.DeepEqual(events[len(events)-len(wantTail):], wantTail) {
				t.Errorf("events = %v, want cleanup tail %v", events, wantTail)
			}
		})
	}
}

func TestNativeModelCloseRetriesInReverseOrder(t *testing.T) {
	events := &eventLog{}
	session := &fakeNativeSession{fakeNativeResource: fakeNativeResource{name: "session", events: events, closeErrs: []error{errors.New("secret close"), nil}}}
	tokenizer := &fakeNativeTokenizer{fakeNativeResource: fakeNativeResource{name: "tokenizer", events: events}}
	source := &fakeNativeSource{events: events, manifest: loadTransformManifest(t), closeErrs: []error{errors.New("secret source close"), nil}}
	model := &nativeModelEngine{session: session, tokenizer: tokenizer, source: source}

	if err := model.Close(t.Context()); !errors.Is(err, ErrNativeFailure) {
		t.Fatalf("first Close error = %v, want errors.Is(_, ErrNativeFailure)", err)
	} else if strings.Contains(err.Error(), "secret") {
		t.Errorf("Close disclosed underlying detail: %v", err)
	}
	if got := events.snapshot(); !reflect.DeepEqual(got, []string{"session.close"}) {
		t.Errorf("events after failed close = %v", got)
	}
	if err := model.Close(t.Context()); !errors.Is(err, ErrMaterialization) {
		t.Fatalf("second Close error = %v, want ErrMaterialization", err)
	} else if strings.Contains(err.Error(), "secret") {
		t.Errorf("source Close disclosed underlying detail: %v", err)
	}
	if got := events.snapshot(); !reflect.DeepEqual(got, []string{"session.close", "session.close", "tokenizer.close", "source.close"}) {
		t.Errorf("events after second close = %v", got)
	}
	if err := model.Close(t.Context()); err != nil {
		t.Fatalf("source close retry returned unexpected error: %v", err)
	}
	wantEvents := []string{"session.close", "session.close", "tokenizer.close", "source.close", "source.close"}
	if got := events.snapshot(); !reflect.DeepEqual(got, wantEvents) {
		t.Errorf("events after source retry = %v, want %v", got, wantEvents)
	}
}

func TestNativeRuntimeEngineFilesystemSourceAndLeaseClose(t *testing.T) {
	coordinator := &nativeCoordinator{}
	environment := &fakeNativeEnvironment{version: nativeONNXRuntimeVersion}
	key := nativeEnvironmentKey{libraryPath: "/runtime/libonnxruntime.so", librarySHA256: "digest"}
	lease, err := coordinator.acquire(t.Context(), key, environment)
	if err != nil {
		t.Fatalf("acquire returned unexpected error: %v", err)
	}
	engine := &nativeRuntimeEngine{lease: lease, factory: newFakeNativeFactory(loadTransformManifest(t))}
	model, err := engine.OpenModelFS(t.Context(), fstestFS{}, FSModelOptions{WorkDir: t.TempDir()})
	if err != nil {
		t.Fatalf("OpenModelFS() returned unexpected error: %v", err)
	}
	if err := model.Close(t.Context()); err != nil {
		t.Fatalf("filesystem model Close() returned unexpected error: %v", err)
	}
	canceled, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := engine.OpenModelFS(canceled, fstestFS{}, FSModelOptions{}); !errors.Is(err, context.Canceled) {
		t.Errorf("OpenModelFS canceled error = %v, want errors.Is(_, context.Canceled)", err)
	}
	if err := engine.Close(t.Context()); err != nil {
		t.Fatalf("Close returned unexpected error: %v", err)
	}
	if err := engine.Close(t.Context()); err != nil {
		t.Fatalf("repeated Close returned unexpected error: %v", err)
	}
	if got := environment.snapshot().destroyCalls; got != 1 {
		t.Errorf("destroy calls = %d, want 1", got)
	}
}

func TestNativeRuntimeFilesystemSourceErrorCategory(t *testing.T) {
	factory := newFakeNativeFactory(loadTransformManifest(t))
	factory.verifyErr = ErrMaterialization
	engine := &nativeRuntimeEngine{factory: factory}
	model, err := engine.OpenModelFS(t.Context(), fstestFS{}, FSModelOptions{WorkDir: t.TempDir()})
	if model != nil {
		t.Fatal("OpenModelFS(materialization failure) returned a non-nil model")
	}
	if !errors.Is(err, ErrMaterialization) {
		t.Errorf("OpenModelFS(materialization failure) error = %v, want ErrMaterialization", err)
	}
	if got := factory.eventSnapshot(); !reflect.DeepEqual(got, []string{"materialize"}) {
		t.Errorf("events = %v, want [materialize]", got)
	}
}

type nativeEnvironmentSnapshot struct {
	path            string
	initializeCalls int
	destroyCalls    int
}

type fakeNativeEnvironment struct {
	mu sync.Mutex

	path            string
	version         string
	initializeErr   error
	destroyErrs     []error
	initializeCalls int
	destroyCalls    int
}

func (e *fakeNativeEnvironment) SetSharedLibraryPath(path string) {
	e.mu.Lock()
	e.path = path
	e.mu.Unlock()
}

func (e *fakeNativeEnvironment) Initialize() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.initializeCalls++
	return e.initializeErr
}

func (e *fakeNativeEnvironment) Destroy() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.destroyCalls++
	if len(e.destroyErrs) == 0 {
		return nil
	}
	err := e.destroyErrs[0]
	e.destroyErrs = e.destroyErrs[1:]
	return err
}

func (e *fakeNativeEnvironment) Version() string {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.version
}

func (e *fakeNativeEnvironment) snapshot() nativeEnvironmentSnapshot {
	e.mu.Lock()
	defer e.mu.Unlock()
	return nativeEnvironmentSnapshot{
		path:            e.path,
		initializeCalls: e.initializeCalls,
		destroyCalls:    e.destroyCalls,
	}
}

type eventLog struct {
	mu     sync.Mutex
	events []string
}

func (l *eventLog) add(event string) {
	l.mu.Lock()
	l.events = append(l.events, event)
	l.mu.Unlock()
}

func (l *eventLog) snapshot() []string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]string(nil), l.events...)
}

type fakeNativeResource struct {
	mu sync.Mutex

	name      string
	events    *eventLog
	closeErrs []error
}

func (r *fakeNativeResource) Close() error {
	r.events.add(r.name + ".close")
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.closeErrs) == 0 {
		return nil
	}
	err := r.closeErrs[0]
	r.closeErrs = r.closeErrs[1:]
	return err
}

type fakeNativeTokenizer struct {
	fakeNativeResource
}

func (*fakeNativeTokenizer) Encode(string) ([]uint32, error) { return []uint32{1}, nil }

type fakeNativeSession struct {
	fakeNativeResource
	run func(context.Context, preparedBatch) (rawModelOutputs, error)
}

func (s *fakeNativeSession) Run(ctx context.Context, batch preparedBatch) (rawModelOutputs, error) {
	if s.run == nil {
		return rawModelOutputs{}, errors.New("fake session run is not configured")
	}
	return s.run(ctx, batch)
}

type fakeNativeFactory struct {
	mu sync.Mutex

	events          eventLog
	manifest        bundle.Manifest
	config          []byte
	verifyErr       error
	readErr         error
	pathErr         error
	pathErrAt       string
	sourceCloseErrs []error
	tokenizerErr    error
	sessionErr      error
	cancel          context.CancelFunc
	cancelStage     string
	inputNames      []string
	outputNames     []string
}

func newFakeNativeFactory(manifest bundle.Manifest) *fakeNativeFactory {
	return &fakeNativeFactory{
		manifest: manifest,
		config:   []byte(`{"temperature":[1,1,1],"temperature_by_options":{}}`),
	}
}

func (f *fakeNativeFactory) OpenDirectorySource(context.Context, string) (nativeBundleSource, error) {
	f.events.add("verify")
	if f.verifyErr != nil {
		return nil, f.verifyErr
	}
	source := f.newSource()
	if f.cancelStage == "source" {
		f.cancel()
	}
	return source, nil
}

func (f *fakeNativeFactory) OpenFilesystemSource(context.Context, fs.FS, FSModelOptions) (nativeBundleSource, error) {
	f.events.add("materialize")
	if f.verifyErr != nil {
		return nil, f.verifyErr
	}
	source := f.newSource()
	if f.cancelStage == "source" {
		f.cancel()
	}
	return source, nil
}

func (f *fakeNativeFactory) newSource() *fakeNativeSource {
	return &fakeNativeSource{
		events:    &f.events,
		manifest:  f.manifest,
		config:    append([]byte(nil), f.config...),
		readErr:   f.readErr,
		pathErr:   f.pathErr,
		pathErrAt: f.pathErrAt,
		closeErrs: append([]error(nil), f.sourceCloseErrs...),
	}
}

func (f *fakeNativeFactory) OpenTokenizer(path string) (nativeTokenizerResource, error) {
	f.events.add("tokenizer:" + path)
	if f.tokenizerErr != nil {
		return nil, f.tokenizerErr
	}
	tokenizer := &fakeNativeTokenizer{fakeNativeResource: fakeNativeResource{name: "tokenizer", events: &f.events}}
	if f.cancelStage == "tokenizer" {
		f.cancel()
	}
	return tokenizer, nil
}

func (f *fakeNativeFactory) OpenSession(
	path string,
	inputNames []string,
	outputNames []string,
) (nativeSessionResource, error) {
	f.events.add("session:" + path)
	f.mu.Lock()
	f.inputNames = append([]string(nil), inputNames...)
	f.outputNames = append([]string(nil), outputNames...)
	f.mu.Unlock()
	if f.sessionErr != nil {
		return nil, f.sessionErr
	}
	session := &fakeNativeSession{fakeNativeResource: fakeNativeResource{name: "session", events: &f.events}}
	if f.cancelStage == "session" {
		f.cancel()
	}
	return session, nil
}

func (f *fakeNativeFactory) eventSnapshot() []string { return f.events.snapshot() }

type fstestFS struct{}

func (fstestFS) Open(string) (fs.File, error) { return nil, fs.ErrNotExist }

type fakeNativeSource struct {
	events    *eventLog
	manifest  bundle.Manifest
	config    []byte
	readErr   error
	pathErr   error
	pathErrAt string
	closeErrs []error
}

func (s *fakeNativeSource) Manifest() bundle.Manifest { return s.manifest }

func (s *fakeNativeSource) Path(path string) (string, error) {
	if s.pathErr != nil && (s.pathErrAt == "" || s.pathErrAt == path) {
		return "", s.pathErr
	}
	return filepath.Join("/verified", filepath.FromSlash(path)), nil
}

func (s *fakeNativeSource) ReadFile(_ context.Context, path string, _ int64) ([]byte, error) {
	s.events.add("read:" + path)
	if s.readErr != nil {
		return nil, s.readErr
	}
	return append([]byte(nil), s.config...), nil
}

func (s *fakeNativeSource) Close() error {
	s.events.add("source.close")
	if len(s.closeErrs) == 0 {
		return nil
	}
	err := s.closeErrs[0]
	s.closeErrs = s.closeErrs[1:]
	return err
}
