package laya

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"sync"
	"testing"
)

func TestNativeModelPredictExecutesPreparedBatch(t *testing.T) {
	corpus := loadTransformCorpus(t)
	manifest := loadTransformManifest(t)
	corpusCases := make(map[string]struct {
		state      State
		question   Question
		inputUsage int
	}, len(corpus.Cases))
	for _, corpusCase := range corpus.Cases {
		corpusCases[corpusCase.ID] = struct {
			state      State
			question   Question
			inputUsage int
		}{
			state:      stateFromCorpus(t, corpusCase.State),
			question:   questionFromCorpus(t, corpusCase),
			inputUsage: len(corpusCase.Sequence.InputIDs),
		}
	}
	run := corpus.Runs[0]
	runCaseIDs := run.CaseIDs[:2]
	questions := make([]Question, len(runCaseIDs))
	state := corpusCases[run.CaseIDs[0]].state
	wantInputUsage := 0
	for index, caseID := range runCaseIDs {
		item := corpusCases[caseID]
		if !reflect.DeepEqual(item.state, state) {
			t.Fatalf("mixed run case %q does not share its State", caseID)
		}
		questions[index] = item.question
		wantInputUsage += item.inputUsage
	}

	tokenizer := &closableTextTokenizer{textTokenizer: newCorpusScriptedTokenizer(t, corpus, manifest)}
	var captured preparedBatch
	session := &fakeNativeSession{
		fakeNativeResource: fakeNativeResource{name: "session", events: &eventLog{}},
		run: func(_ context.Context, batch preparedBatch) (rawModelOutputs, error) {
			captured = batch
			return rawModelOutputs{
				logits:       float32Rows(run.ONNXOutputs.Logits[:len(runCaseIDs)]),
				actionLogits: float32Rows(run.ONNXOutputs.ActLogits[:len(runCaseIDs)]),
			}, nil
		},
	}
	calibration, err := parseCalibrationConfig(
		manifest,
		[]byte(`{"temperature":[1,1,1],"temperature_by_options":{}}`),
	)
	if err != nil {
		t.Fatalf("parse calibration: %v", err)
	}
	engine := &nativeModelEngine{
		tokenizer:   tokenizer,
		session:     session,
		manifest:    manifest,
		calibration: calibration,
		runtimeID:   "runtime-test",
	}

	prediction, err := engine.Predict(t.Context(), state, questions)
	if err != nil {
		t.Fatalf("Predict returned unexpected error: %v", err)
	}
	wantTensors := run.Tensors
	wantTensors.InputIDs = wantTensors.InputIDs[:len(runCaseIDs)]
	wantTensors.AttentionMask = wantTensors.AttentionMask[:len(runCaseIDs)]
	wantTensors.MarkerPos = wantTensors.MarkerPos[:len(runCaseIDs)]
	wantTensors.MarkerMask = wantTensors.MarkerMask[:len(runCaseIDs)]
	wantTensors.QType = wantTensors.QType[:len(runCaseIDs)]
	assertPreparedTensors(t, captured, wantTensors)
	if prediction.Usage() != (Usage{InputTokens: wantInputUsage}) {
		t.Errorf("prediction usage = %+v, want input tokens %d", prediction.Usage(), wantInputUsage)
	}
	metadata := prediction.Metadata()
	if metadata.ModelID != ModelID(manifest.Bundle.ID) || metadata.BundleID != BundleID(manifest.ID()) || metadata.RuntimeID != "runtime-test" {
		t.Errorf("prediction metadata = %+v", metadata)
	}
	results := prediction.Results()
	for index, caseID := range runCaseIDs {
		var corpusCaseFound bool
		for _, corpusCase := range corpus.Cases {
			if corpusCase.ID == caseID {
				assertTypedResult(t, results[index], captured.items[index].question, corpusCase)
				corpusCaseFound = true
				break
			}
		}
		if !corpusCaseFound {
			t.Fatalf("corpus case %q was not found", caseID)
		}
	}
}

func TestNativeModelPredictFailureCategories(t *testing.T) {
	manifest := loadTransformManifest(t)
	question, err := NewNoulQuestion("valid", "Is it valid?", "no", "yes")
	if err != nil {
		t.Fatalf("NewNoulQuestion returned unexpected error: %v", err)
	}
	calibration, err := parseCalibrationConfig(
		manifest,
		[]byte(`{"temperature":[1,1,1],"temperature_by_options":{}}`),
	)
	if err != nil {
		t.Fatalf("parse calibration: %v", err)
	}
	tests := []struct {
		name      string
		run       func(context.Context, preparedBatch) (rawModelOutputs, error)
		wantError error
	}{
		{
			name: "native run",
			run: func(context.Context, preparedBatch) (rawModelOutputs, error) {
				return rawModelOutputs{}, errors.New("secret tensor content")
			},
			wantError: ErrNativeFailure,
		},
		{
			name: "invalid output",
			run: func(context.Context, preparedBatch) (rawModelOutputs, error) {
				return rawModelOutputs{logits: [][]float32{{1}}, actionLogits: [][]float32{{1, 2}}}, nil
			},
			wantError: ErrInvalidOutput,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tokenizer := &closableTextTokenizer{textTokenizer: fixedTokenizer{ids: []uint32{1}}}
			session := &fakeNativeSession{
				fakeNativeResource: fakeNativeResource{name: "session", events: &eventLog{}},
				run:                test.run,
			}
			engine := &nativeModelEngine{
				tokenizer:   tokenizer,
				session:     session,
				manifest:    manifest,
				calibration: calibration,
			}
			prediction, err := engine.Predict(t.Context(), TextState("state"), []Question{question})
			if prediction.Results() != nil {
				t.Errorf("Predict results = %v, want nil", prediction.Results())
			}
			if !errors.Is(err, test.wantError) {
				t.Errorf("Predict error = %v, want errors.Is(_, %v)", err, test.wantError)
			}
			if strings.Contains(err.Error(), "secret") {
				t.Errorf("Predict error disclosed underlying content: %v", err)
			}
		})
	}
}

func TestNativeModelPredictHonorsPreRunCancellation(t *testing.T) {
	manifest := loadTransformManifest(t)
	question, err := NewNoulQuestion("valid", "Is it valid?", "no", "yes")
	if err != nil {
		t.Fatalf("NewNoulQuestion returned unexpected error: %v", err)
	}
	var runCalls int
	session := &fakeNativeSession{
		fakeNativeResource: fakeNativeResource{name: "session", events: &eventLog{}},
		run: func(context.Context, preparedBatch) (rawModelOutputs, error) {
			runCalls++
			return rawModelOutputs{}, nil
		},
	}
	engine := &nativeModelEngine{
		tokenizer: &closableTextTokenizer{textTokenizer: fixedTokenizer{ids: []uint32{1}}},
		session:   session,
		manifest:  manifest,
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := engine.Predict(ctx, TextState("state"), []Question{question}); !errors.Is(err, context.Canceled) {
		t.Errorf("Predict error = %v, want errors.Is(_, context.Canceled)", err)
	}
	if runCalls != 0 {
		t.Errorf("session run calls = %d, want 0", runCalls)
	}
}

func TestNativeModelPredictDiscardsCanceledRunAndRemainsUsable(t *testing.T) {
	corpus := loadTransformCorpus(t)
	manifest := loadTransformManifest(t)
	corpusCase := corpus.Cases[0]
	state := stateFromCorpus(t, corpusCase.State)
	question := questionFromCorpus(t, corpusCase)
	calibration, err := parseCalibrationConfig(
		manifest,
		[]byte(`{"temperature":[1,1,1],"temperature_by_options":{}}`),
	)
	if err != nil {
		t.Fatalf("parse calibration: %v", err)
	}
	started := make(chan struct{})
	secretCause := errors.New("secret terminated native call")
	runCalls := 0
	session := &fakeNativeSession{
		fakeNativeResource: fakeNativeResource{name: "session", events: &eventLog{}},
		run: func(ctx context.Context, batch preparedBatch) (rawModelOutputs, error) {
			runCalls++
			if runCalls == 1 {
				close(started)
				<-ctx.Done()
				return rawModelOutputs{}, nativeRunCancellationError(ctx.Err(), secretCause)
			}
			markerWidth := len(batch.markerPos[0])
			return rawModelOutputs{
				logits:       [][]float32{float32Values(corpusCase.Raw.Logits[:markerWidth])},
				actionLogits: [][]float32{float32Values(corpusCase.Raw.ActLogits)},
			}, nil
		},
	}
	engine := &nativeModelEngine{
		tokenizer:   &closableTextTokenizer{textTokenizer: newCorpusScriptedTokenizer(t, corpus, manifest)},
		session:     session,
		manifest:    manifest,
		calibration: calibration,
	}
	ctx, cancel := context.WithCancel(t.Context())
	done := make(chan error, 1)
	go func() {
		prediction, predictErr := engine.Predict(ctx, state, []Question{question})
		if prediction.Results() != nil {
			done <- errors.New("canceled prediction published results")
			return
		}
		done <- predictErr
	}()
	<-started
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) || !errors.Is(err, secretCause) {
		t.Errorf("canceled Predict error = %v, want context and native causes", err)
	} else if strings.Contains(err.Error(), "secret") {
		t.Errorf("canceled Predict disclosed underlying detail: %v", err)
	}

	prediction, err := engine.Predict(t.Context(), state, []Question{question})
	if err != nil {
		t.Fatalf("Predict after cancellation returned unexpected error: %v", err)
	}
	if len(prediction.Results()) != 1 || runCalls != 2 {
		t.Errorf("post-cancel prediction results/calls = %d/%d, want 1/2", len(prediction.Results()), runCalls)
	}
}

type closableTextTokenizer struct {
	mu sync.Mutex

	textTokenizer
	closeCalls int
}

func (t *closableTextTokenizer) Close() error {
	t.mu.Lock()
	t.closeCalls++
	t.mu.Unlock()
	return nil
}

type fixedTokenizer struct {
	ids []uint32
}

func (t fixedTokenizer) Encode(string) ([]uint32, error) {
	return append([]uint32(nil), t.ids...), nil
}

func float32Values(values []float64) []float32 {
	converted := make([]float32, len(values))
	for index, value := range values {
		converted[index] = float32(value)
	}
	return converted
}
