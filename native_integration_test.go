//go:build laya_native && cgo

package laya

import (
	"errors"
	"math"
	"os"
	"testing"

	ort "github.com/yalue/onnxruntime_go"
)

func TestNativeRuntimeOpensVerifiedLocalBundle(t *testing.T) {
	bundleDir := os.Getenv("LAYA_BUNDLE_DIR")
	if bundleDir == "" || os.Getenv(nativeLibraryEnvironment) == "" {
		t.Skip("protected native artifacts were not supplied")
	}
	runProtectedModelParity(t, func(runtime *Runtime) (*Model, error) {
		return runtime.OpenModelDir(t.Context(), bundleDir, ModelOptions{Truncation: TruncateOverflow})
	})
}

func runProtectedModelParity(t *testing.T, open func(*Runtime) (*Model, error)) {
	t.Helper()
	layaRuntime, err := NewRuntime(t.Context(), RuntimeOptions{})
	if err != nil {
		t.Fatalf("NewRuntime returned unexpected error: %v", err)
	}
	model, err := open(layaRuntime)
	if err != nil {
		_ = layaRuntime.Close(t.Context())
		t.Fatalf("open protected model returned unexpected error: %v", err)
	}
	corpus := loadTransformCorpus(t)
	manifest := loadTransformManifest(t)
	cases := make(map[string]struct {
		state    State
		question Question
	}, len(corpus.Cases))
	preparedItems := make(map[string]preparedItem, len(corpus.Cases))
	nativeEngine := model.engine.(*nativeModelEngine)
	for _, corpusCase := range corpus.Cases {
		state := stateFromCorpus(t, corpusCase.State)
		question := questionFromCorpus(t, corpusCase)
		prediction, predictErr := model.Predict(t.Context(), state, []Question{question})
		if predictErr != nil {
			t.Fatalf("Predict corpus case %q returned unexpected error: %v", corpusCase.ID, predictErr)
		}
		results := prediction.Results()
		if len(results) != 1 {
			t.Fatalf("Predict corpus case %q result count = %d, want 1", corpusCase.ID, len(results))
		}
		prepared, prepareErr := normalizeQuestion(question, manifest)
		if prepareErr != nil {
			t.Fatalf("normalize corpus case %q: %v", corpusCase.ID, prepareErr)
		}
		assertTypedResult(t, results[0], prepared, corpusCase)
		if prediction.Usage() != (Usage{InputTokens: len(corpusCase.Sequence.InputIDs)}) {
			t.Errorf("corpus case %q usage = %+v", corpusCase.ID, prediction.Usage())
		}
		metadata := prediction.Metadata()
		if metadata.ModelID != ModelID(manifest.Bundle.ID) ||
			metadata.BundleID != BundleID(corpus.Metadata.BundleID) || metadata.RuntimeID == "" {
			t.Errorf("corpus case %q metadata = %+v", corpusCase.ID, metadata)
		}
		if metadata.Truncation.Truncated != corpusCase.Sequence.Truncated ||
			metadata.Truncation.OriginalInputTokens != corpusCase.Sequence.OriginalInputTokens ||
			metadata.Truncation.EffectiveInputTokens != corpusCase.Sequence.EffectiveInputTokens {
			t.Errorf("corpus case %q truncation = %+v", corpusCase.ID, metadata.Truncation)
		}
		cases[corpusCase.ID] = struct {
			state    State
			question Question
		}{state: state, question: question}
		batch, prepareErr := prepareInferenceBatch(
			state,
			[]Question{question},
			manifest,
			nativeEngine.tokenizer,
			ModelOptions{Truncation: TruncateOverflow},
		)
		if prepareErr != nil {
			t.Fatalf("prepare native corpus case %q: %v", corpusCase.ID, prepareErr)
		}
		preparedItems[corpusCase.ID] = batch.items[0]
	}
	for _, corpusRun := range corpus.Runs {
		items := make([]preparedItem, len(corpusRun.CaseIDs))
		for index, caseID := range corpusRun.CaseIDs {
			items[index] = preparedItems[caseID]
		}
		batch := collatePrepared(items, manifest.Tokenizer.SpecialTokens.Pad)
		assertPreparedTensors(t, batch, corpusRun.Tensors)
		outputs, runErr := nativeEngine.session.Run(t.Context(), batch)
		if runErr != nil {
			t.Fatalf("run native parity batch %q: %v", corpusRun.ID, runErr)
		}
		assertNativeRowsClose(t, corpusRun.ID+" logits", outputs.logits, corpusRun.ONNXOutputs.Logits)
		assertNativeRowsClose(t, corpusRun.ID+" action logits", outputs.actionLogits, corpusRun.ONNXOutputs.ActLogits)
	}

	mixedRun := corpus.Runs[0]
	mixedCaseIDs := mixedRun.CaseIDs[:2]
	mixedQuestions := make([]Question, len(mixedCaseIDs))
	for index, caseID := range mixedCaseIDs {
		mixedQuestions[index] = cases[caseID].question
	}
	mixed, err := model.Predict(t.Context(), cases[mixedCaseIDs[0]].state, mixedQuestions)
	if err != nil {
		t.Fatalf("Predict ordered mixed batch returned unexpected error: %v", err)
	}
	if got := mixed.Results(); len(got) != len(mixedCaseIDs) ||
		got[0].QuestionID() != mixedQuestions[0].ID() || got[1].QuestionID() != mixedQuestions[1].ID() {
		t.Fatalf("ordered mixed results = %v", got)
	}
	if err := model.Close(t.Context()); err != nil {
		t.Fatalf("model Close returned unexpected error: %v", err)
	}
	if err := layaRuntime.Close(t.Context()); err != nil {
		t.Fatalf("runtime Close returned unexpected error: %v", err)
	}
}

func assertNativeRowsClose(t *testing.T, name string, got [][]float32, want [][]float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s rows = %d, want %d", name, len(got), len(want))
	}
	for row := range got {
		if len(got[row]) != len(want[row]) {
			t.Fatalf("%s row %d width = %d, want %d", name, row, len(got[row]), len(want[row]))
		}
		for column := range got[row] {
			actual := float64(got[row][column])
			expected := want[row][column]
			limit := 1e-4 + 1e-4*math.Abs(expected)
			if math.Abs(actual-expected) > limit {
				t.Fatalf(
					"%s[%d][%d] = %.9g, want %.9g within %.9g",
					name,
					row,
					column,
					actual,
					expected,
					limit,
				)
			}
		}
	}
}

func TestNativeOutputContractRejectsMalformedTensors(t *testing.T) {
	if os.Getenv(nativeLibraryEnvironment) == "" {
		t.Skip("protected native artifacts were not supplied")
	}
	layaRuntime, err := NewRuntime(t.Context(), RuntimeOptions{})
	if err != nil {
		t.Fatalf("NewRuntime returned unexpected error: %v", err)
	}
	defer func() {
		if err := layaRuntime.Close(t.Context()); err != nil {
			t.Errorf("runtime Close returned unexpected error: %v", err)
		}
	}()

	floatTensor := func(shape ort.Shape, data []float32) ort.Value {
		t.Helper()
		value, tensorErr := ort.NewTensor(shape, data)
		if tensorErr != nil {
			t.Fatalf("create float tensor: %v", tensorErr)
		}
		return value
	}
	intTensor := func(shape ort.Shape, data []int64) ort.Value {
		t.Helper()
		value, tensorErr := ort.NewTensor(shape, data)
		if tensorErr != nil {
			t.Fatalf("create int tensor: %v", tensorErr)
		}
		return value
	}
	tests := []struct {
		name   string
		values func() []ort.Value
	}{
		{name: "count", values: func() []ort.Value { return nil }},
		{
			name: "dtype",
			values: func() []ort.Value {
				return []ort.Value{
					intTensor(ort.NewShape(1, 2), []int64{1, 2}),
					floatTensor(ort.NewShape(1, 2), []float32{1, 2}),
				}
			},
		},
		{
			name: "rank",
			values: func() []ort.Value {
				return []ort.Value{
					floatTensor(ort.NewShape(2), []float32{1, 2}),
					floatTensor(ort.NewShape(1, 2), []float32{1, 2}),
				}
			},
		},
		{
			name: "dimension",
			values: func() []ort.Value {
				return []ort.Value{
					floatTensor(ort.NewShape(1, 3), []float32{1, 2, 3}),
					floatTensor(ort.NewShape(1, 2), []float32{1, 2}),
				}
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			values := test.values()
			defer func() {
				if err := destroyORTValues(values); err != nil {
					t.Errorf("destroy malformed values: %v", err)
				}
			}()
			if _, err := copyORTOutputs(values, 1, 2); !errors.Is(err, ErrInvalidOutput) {
				t.Errorf("copyORTOutputs error = %v, want errors.Is(_, ErrInvalidOutput)", err)
			}
		})
	}
}
