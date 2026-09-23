package laya

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/metalagman/laya-go/internal/bundle"
	"github.com/metalagman/laya-go/internal/parity"
)

func TestCorpusBackedInferenceTransforms(t *testing.T) {
	corpus := loadTransformCorpus(t)
	manifest := loadTransformManifest(t)
	tokenizer := newCorpusScriptedTokenizer(t, corpus, manifest)
	calibration, err := parseCalibrationConfig(manifest, []byte(`{
  "temperature": [1.0, 1.0, 1.0],
  "temperature_by_options": {}
}`))
	if err != nil {
		t.Fatalf("parse calibration config: %v", err)
	}

	items := make(map[string]preparedItem, len(corpus.Cases))
	cases := make(map[string]parity.Case, len(corpus.Cases))
	for _, corpusCase := range corpus.Cases {
		corpusCase := corpusCase
		t.Run("prepare/"+corpusCase.ID, func(t *testing.T) {
			state := stateFromCorpus(t, corpusCase.State)
			question := questionFromCorpus(t, corpusCase)
			policy := RejectOverflow
			if corpusCase.Sequence.Truncated {
				policy = TruncateOverflow
			}
			batch, err := prepareInferenceBatch(
				state,
				[]Question{question},
				manifest,
				tokenizer,
				ModelOptions{Truncation: policy},
			)
			if err != nil {
				t.Fatalf("prepare corpus case: %v", err)
			}
			if len(batch.items) != 1 {
				t.Fatalf("prepared item count = %d, want 1", len(batch.items))
			}
			item := batch.items[0]
			if !reflect.DeepEqual(item.question.options, corpusCase.RenderedOptions) {
				t.Fatalf("rendered options = %#v, want %#v", item.question.options, corpusCase.RenderedOptions)
			}
			if !reflect.DeepEqual(item.inputIDs, corpusCase.Sequence.InputIDs) {
				t.Fatalf("input IDs differ\n got: %v\nwant: %v", item.inputIDs, corpusCase.Sequence.InputIDs)
			}
			if !reflect.DeepEqual(item.markerPositions, corpusCase.Sequence.MarkerPositions) {
				t.Fatalf("marker positions = %v, want %v", item.markerPositions, corpusCase.Sequence.MarkerPositions)
			}
			if item.question.qtype != int64(corpusCase.Sequence.QType) {
				t.Fatalf("qtype = %d, want %d", item.question.qtype, corpusCase.Sequence.QType)
			}
			if item.originalTokens != corpusCase.Sequence.OriginalInputTokens ||
				len(item.inputIDs) != corpusCase.Sequence.EffectiveInputTokens ||
				item.truncated != corpusCase.Sequence.Truncated {
				t.Fatalf(
					"sequence metadata = (%d,%d,%t), want (%d,%d,%t)",
					item.originalTokens,
					len(item.inputIDs),
					item.truncated,
					corpusCase.Sequence.OriginalInputTokens,
					corpusCase.Sequence.EffectiveInputTokens,
					corpusCase.Sequence.Truncated,
				)
			}
			items[corpusCase.ID] = item
			cases[corpusCase.ID] = corpusCase
		})
	}

	for _, corpusRun := range corpus.Runs {
		corpusRun := corpusRun
		t.Run("batch/"+corpusRun.ID, func(t *testing.T) {
			runItems := make([]preparedItem, len(corpusRun.CaseIDs))
			for index, caseID := range corpusRun.CaseIDs {
				runItems[index] = items[caseID]
			}
			batch := collatePrepared(runItems, manifest.Tokenizer.SpecialTokens.Pad)
			assertPreparedTensors(t, batch, corpusRun.Tensors)
			if batch.usage.InputTokens != corpusRun.Usage.InputTokens || batch.usage.OutputTokens != corpusRun.Usage.OutputTokens {
				t.Fatalf("usage = %+v, want %+v", batch.usage, corpusRun.Usage)
			}

			outputs := rawModelOutputs{
				logits:       float32Rows(corpusRun.ReferenceOutputs.Logits),
				actionLogits: float32Rows(corpusRun.ReferenceOutputs.ActLogits),
			}
			metadata := PredictionMetadata{
				ModelID:   "model-test",
				BundleID:  "bundle-test",
				RuntimeID: "runtime-test",
			}
			prediction, err := formatInferencePrediction(batch, outputs, calibration, metadata)
			if err != nil {
				t.Fatalf("format corpus run: %v", err)
			}
			if prediction.Usage() != batch.usage {
				t.Fatalf("prediction usage = %+v, want %+v", prediction.Usage(), batch.usage)
			}
			if got := prediction.Metadata(); got.ModelID != metadata.ModelID || got.BundleID != metadata.BundleID ||
				got.RuntimeID != metadata.RuntimeID || got.Truncation != batch.truncation {
				t.Fatalf("prediction metadata = %+v, want identities %+v and truncation %+v", got, metadata, batch.truncation)
			}
			results := prediction.Results()
			for index, caseID := range corpusRun.CaseIDs {
				corpusCase := cases[caseID]
				assertDerivedRow(t, batch.items[index].question, outputs, index, calibration, corpusCase)
				assertTypedResult(t, results[index], batch.items[index].question, corpusCase)
			}
		})
	}
}

func TestCorpusBoundariesRejectWithoutSilentMutation(t *testing.T) {
	corpus := loadTransformCorpus(t)
	manifest := loadTransformManifest(t)
	tokenizer := newCorpusScriptedTokenizer(t, corpus, manifest)
	cases := make(map[string]parity.Case, len(corpus.Cases))
	for _, corpusCase := range corpus.Cases {
		cases[corpusCase.ID] = corpusCase
	}

	overflow := cases["right-truncated-choice"]
	_, err := prepareInferenceBatch(
		stateFromCorpus(t, overflow.State),
		[]Question{questionFromCorpus(t, overflow)},
		manifest,
		tokenizer,
		ModelOptions{},
	)
	assertInputTooLong(t, err, overflow.Sequence.OriginalInputTokens, manifest.Preprocessing.MaxLen)

	criteria := make([]ChoiceCriterion, 300)
	for index := range criteria {
		criteria[index] = ChoiceCriterion{
			ID:          CriterionID(fmt.Sprintf("option-%03d", index)),
			Description: fmt.Sprintf("value %d", index),
		}
	}
	impossible, err := NewChoiceQuestion("impossible", "Too many options", criteria)
	if err != nil {
		t.Fatalf("construct impossible question: %v", err)
	}
	_, err = prepareInferenceBatch(
		TextState("marker safety"),
		[]Question{impossible},
		manifest,
		tokenizer,
		ModelOptions{Truncation: TruncateOverflow},
	)
	if !errors.Is(err, ErrInputTooLong) {
		t.Fatalf("impossible marker layout error = %v, want ErrInputTooLong", err)
	}
}

func TestInputTokenLimitLowersBundleLimit(t *testing.T) {
	corpus := loadTransformCorpus(t)
	manifest := loadTransformManifest(t)
	tokenizer := newCorpusScriptedTokenizer(t, corpus, manifest)
	var exact parity.Case
	for _, corpusCase := range corpus.Cases {
		if corpusCase.ID == "exact-fit-choice" {
			exact = corpusCase
			break
		}
	}
	const lowerLimit = 1000
	_, err := prepareInferenceBatch(
		stateFromCorpus(t, exact.State),
		[]Question{questionFromCorpus(t, exact)},
		manifest,
		tokenizer,
		ModelOptions{InputTokenLimit: lowerLimit},
	)
	assertInputTooLong(t, err, exact.Sequence.OriginalInputTokens, lowerLimit)

	batch, err := prepareInferenceBatch(
		stateFromCorpus(t, exact.State),
		[]Question{questionFromCorpus(t, exact)},
		manifest,
		tokenizer,
		ModelOptions{InputTokenLimit: lowerLimit, Truncation: TruncateOverflow},
	)
	if err != nil {
		t.Fatalf("truncate to lower input limit: %v", err)
	}
	if len(batch.items[0].inputIDs) != lowerLimit || batch.truncation != (TruncationMetadata{
		Truncated:            true,
		OriginalInputTokens:  exact.Sequence.OriginalInputTokens,
		EffectiveInputTokens: lowerLimit,
	}) {
		t.Fatalf("lower-limit truncation = len %d metadata %+v", len(batch.items[0].inputIDs), batch.truncation)
	}
}

func TestCalibrationSelectionAndClamping(t *testing.T) {
	manifest := loadTransformManifest(t)
	config, err := parseCalibrationConfig(manifest, []byte(`{
  "temperature": [0.1, 6, "invalid"],
  "temperature_by_options": {
    "choice:3-5": "2.5",
    "noul:2": false
  }
}`))
	if err != nil {
		t.Fatalf("parse calibration: %v", err)
	}
	tests := []struct {
		name    string
		kind    preparedQuestionKind
		qtype   int64
		options int
		want    float64
	}{
		{name: "choice qtype lower clamp", kind: preparedChoice, qtype: 0, options: 2, want: 0.5},
		{name: "choice bucket first", kind: preparedChoice, qtype: 0, options: 4, want: 2.5},
		{name: "score upper clamp", kind: preparedScore, qtype: 1, options: 3, want: 5},
		{name: "invalid fallback", kind: preparedNoul, qtype: 2, options: 3, want: 1},
		{name: "boolean bucket then clamp", kind: preparedNoul, qtype: 2, options: 2, want: 0.5},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			value, err := config.selectValue(preparedQuestion{
				kind:    test.kind,
				qtype:   test.qtype,
				options: make([]string, test.options),
			})
			if err != nil {
				t.Fatalf("select calibration: %v", err)
			}
			if value.effective != test.want {
				t.Fatalf("effective temperature = %v, want %v", value.effective, test.want)
			}
		})
	}
}

func TestInvalidRawOutputsFailBeforeFormatting(t *testing.T) {
	corpus := loadTransformCorpus(t)
	manifest := loadTransformManifest(t)
	tokenizer := newCorpusScriptedTokenizer(t, corpus, manifest)
	corpusCase := corpus.Cases[0]
	batch, err := prepareInferenceBatch(
		stateFromCorpus(t, corpusCase.State),
		[]Question{questionFromCorpus(t, corpusCase)},
		manifest,
		tokenizer,
		ModelOptions{},
	)
	if err != nil {
		t.Fatalf("prepare valid batch: %v", err)
	}
	valid := rawModelOutputs{
		logits:       [][]float32{{1, 2, 3}},
		actionLogits: [][]float32{{1, 2}},
	}
	tests := []struct {
		name   string
		mutate func(rawModelOutputs) rawModelOutputs
	}{
		{name: "missing logits row", mutate: func(value rawModelOutputs) rawModelOutputs { value.logits = nil; return value }},
		{name: "logits width", mutate: func(value rawModelOutputs) rawModelOutputs { value.logits[0] = value.logits[0][:2]; return value }},
		{name: "action width", mutate: func(value rawModelOutputs) rawModelOutputs {
			value.actionLogits[0] = value.actionLogits[0][:1]
			return value
		}},
		{name: "non-finite logits", mutate: func(value rawModelOutputs) rawModelOutputs { value.logits[0][1] = float32(math.NaN()); return value }},
		{name: "non-finite actions", mutate: func(value rawModelOutputs) rawModelOutputs {
			value.actionLogits[0][0] = float32(math.Inf(1))
			return value
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			outputs := cloneRawOutputs(valid)
			err := validateRawOutputs(batch, test.mutate(outputs))
			if !errors.Is(err, ErrInvalidOutput) {
				t.Fatalf("validate raw outputs error = %v, want ErrInvalidOutput", err)
			}
		})
	}
}

func TestTokenizerFailureIsTypedAndContentSafe(t *testing.T) {
	manifest := loadTransformManifest(t)
	question, err := NewNoulQuestion("safe-error", "contains private text", "", "")
	if err != nil {
		t.Fatalf("construct question: %v", err)
	}
	cause := errors.New("SECRET model input")
	_, err = prepareInferenceBatch(
		TextState("SECRET state"),
		[]Question{question},
		manifest,
		failingTokenizer{err: cause},
		ModelOptions{},
	)
	if !errors.Is(err, ErrNativeFailure) || !errors.Is(err, cause) {
		t.Fatalf("tokenizer failure = %v, want native category and preserved cause", err)
	}
	if strings.Contains(err.Error(), "SECRET") || strings.Contains(err.Error(), "private") {
		t.Fatalf("tokenizer failure leaks input content: %q", err)
	}
}

func TestParseCalibrationRejectsMalformedOrUnknownConfig(t *testing.T) {
	manifest := loadTransformManifest(t)
	tests := []string{
		`{"temperature":[1,1],"temperature_by_options":{}}`,
		`{"temperature":[1,1,1],"temperature_by_options":{},"unknown":true}`,
		`{"temperature":[1,1,1],"temperature":[2,2,2],"temperature_by_options":{}}`,
		`{"temperature":[1,1,1],"temperature_by_options":{}} trailing`,
	}
	for _, input := range tests {
		if _, err := parseCalibrationConfig(manifest, []byte(input)); err == nil {
			t.Errorf("parseCalibrationConfig(%q) succeeded", input)
		}
	}
}

func FuzzStableSoftmax(f *testing.F) {
	f.Add(float32(-1), float32(0), float32(1), 1.0)
	f.Add(float32(1000), float32(1001), float32(999), 0.5)
	f.Fuzz(func(t *testing.T, first, second, third float32, temperature float64) {
		probabilities, err := stableSoftmax([]float32{first, second, third}, temperature)
		if !isFinite(float64(first)) || !isFinite(float64(second)) || !isFinite(float64(third)) ||
			!isFinite(temperature) || temperature <= 0 {
			if err == nil {
				t.Fatal("invalid softmax inputs unexpectedly succeeded")
			}
			return
		}
		if temperature < 0.5 || temperature > 5 {
			return
		}
		if err != nil {
			t.Fatalf("stableSoftmax returned an unexpected error: %v", err)
		}
		total := 0.0
		for _, probability := range probabilities {
			if !isFinite(probability) || probability < 0 || probability > 1 {
				t.Fatalf("invalid probability %v", probability)
			}
			total += probability
		}
		if math.Abs(total-1) > 1e-12 {
			t.Fatalf("probability total = %.17g, want 1", total)
		}
	})
}

type corpusScriptedTokenizer struct {
	testing *testing.T
	values  map[string][]uint32
}

func newCorpusScriptedTokenizer(t *testing.T, corpus parity.Corpus, manifest bundle.Manifest) *corpusScriptedTokenizer {
	t.Helper()
	tokenizer := &corpusScriptedTokenizer{testing: t, values: make(map[string][]uint32)}
	for _, corpusCase := range corpus.Cases {
		inputIDs := corpusCase.Sequence.InputIDs
		markers := corpusCase.Sequence.MarkerPositions
		if len(markers) == 0 || markers[0] < 2 {
			t.Fatalf("case %s has no usable markers", corpusCase.ID)
		}
		headEnd := int(markers[0]) - 1
		kind := corpusCase.Question.Type
		headText := kind + " question: " + strings.ReplaceAll(corpusCase.Question.Instructions, maskTokenText, " ")
		tokenizer.add(headText, int64ToUint32(t, inputIDs[1:headEnd]))

		lastOptionEnd := -1
		for index := int(markers[len(markers)-1]) + 1; index < len(inputIDs)-1; index++ {
			if inputIDs[index] == int64(manifest.Tokenizer.SpecialTokens.SEP) {
				lastOptionEnd = index
				break
			}
		}
		if lastOptionEnd < 0 {
			t.Fatalf("case %s has no option separator", corpusCase.ID)
		}
		for index, option := range corpusCase.RenderedOptions {
			start := int(markers[index]) + 1
			end := lastOptionEnd
			if index+1 < len(markers) {
				end = int(markers[index+1])
			}
			ids := int64ToUint32(t, inputIDs[start:end])
			if corpusCase.ID == "long-options-choice" {
				ids = append(ids, 900001, 900002, 900003)
			}
			tokenizer.add(" "+strings.ReplaceAll(option, maskTokenText, " "), ids)
		}

		stateText := strings.ReplaceAll(corpusCase.State.Value, maskTokenText, " ")
		stateIDs := int64ToUint32(t, inputIDs[lastOptionEnd+1:len(inputIDs)-1])
		fullStateLength := corpusCase.Sequence.OriginalInputTokens - (lastOptionEnd + 1) - 1
		for len(stateIDs) < fullStateLength {
			stateIDs = append(stateIDs, uint32(800000+len(stateIDs)))
		}
		tokenizer.add(stateText, stateIDs)
	}
	return tokenizer
}

func (t *corpusScriptedTokenizer) Encode(text string) ([]uint32, error) {
	if value, ok := t.values[text]; ok {
		return append([]uint32(nil), value...), nil
	}
	// Unknown strings are used by negative-boundary tests. One token per rune
	// makes the fake deterministic while ensuring large option sets exercise
	// the real head-budget and marker-loss path.
	value := make([]uint32, 0, len([]rune(text)))
	for index := range []rune(text) {
		value = append(value, uint32(700000+index))
	}
	return value, nil
}

func (t *corpusScriptedTokenizer) add(text string, ids []uint32) {
	t.testing.Helper()
	if previous, ok := t.values[text]; ok && !reflect.DeepEqual(previous, ids) {
		t.testing.Fatalf("scripted tokenizer text %q has inconsistent IDs", text)
	}
	t.values[text] = append([]uint32(nil), ids...)
}

type failingTokenizer struct{ err error }

func (t failingTokenizer) Encode(string) ([]uint32, error) { return nil, t.err }

func loadTransformCorpus(t *testing.T) parity.Corpus {
	t.Helper()
	data, err := os.ReadFile("internal/parity/testdata/corpus-v1.json")
	if err != nil {
		t.Fatalf("read parity corpus: %v", err)
	}
	corpus, err := parity.Load(data)
	if err != nil {
		t.Fatalf("load parity corpus: %v", err)
	}
	return corpus
}

func loadTransformManifest(t *testing.T) bundle.Manifest {
	t.Helper()
	data, err := os.ReadFile("internal/bundle/testdata/embedded/manifest.json")
	if err != nil {
		t.Fatalf("read test manifest: %v", err)
	}
	manifest, err := bundle.Parse(data)
	if err != nil {
		t.Fatalf("parse test manifest: %v", err)
	}
	return manifest
}

func stateFromCorpus(t *testing.T, state parity.State) State {
	t.Helper()
	switch state.Kind {
	case "text":
		return TextState(state.Value)
	case "json":
		value, err := JSONState([]byte(state.Value))
		if err != nil {
			t.Fatalf("construct JSON state: %v", err)
		}
		return value
	default:
		t.Fatalf("unknown corpus state kind %q", state.Kind)
		return State{}
	}
}

func questionFromCorpus(t *testing.T, corpusCase parity.Case) Question {
	t.Helper()
	id := QuestionID(corpusCase.ID)
	switch corpusCase.Question.Type {
	case "choice":
		criteria := make([]ChoiceCriterion, len(corpusCase.RenderedOptions))
		for index, option := range corpusCase.RenderedOptions {
			criterionID, description, ok := strings.Cut(option, ": ")
			if !ok {
				t.Fatalf("choice option %q has no separator", option)
			}
			criteria[index] = ChoiceCriterion{ID: CriterionID(criterionID), Description: description}
		}
		question, err := NewChoiceQuestion(id, corpusCase.Question.Instructions, criteria)
		if err != nil {
			t.Fatalf("construct choice question: %v", err)
		}
		return question
	case "score":
		var labels []string
		if err := json.Unmarshal(corpusCase.Question.Criteria, &labels); err != nil {
			t.Fatalf("decode score criteria: %v", err)
		}
		rubric := make([]ScoreLevel, len(labels))
		for index, label := range labels {
			rubric[index] = ScoreLevel{Label: label, Description: label}
		}
		question, err := NewScoreQuestion(id, corpusCase.Question.Instructions, rubric)
		if err != nil {
			t.Fatalf("construct score question: %v", err)
		}
		return question
	case "noul":
		var criteria map[string]string
		if err := json.Unmarshal(corpusCase.Question.Criteria, &criteria); err != nil {
			t.Fatalf("decode noul criteria: %v", err)
		}
		question, err := NewNoulQuestion(
			id,
			corpusCase.Question.Instructions,
			criteria["false"],
			criteria["true"],
		)
		if err != nil {
			t.Fatalf("construct noul question: %v", err)
		}
		return question
	default:
		t.Fatalf("unknown corpus question type %q", corpusCase.Question.Type)
		return nil
	}
}

func int64ToUint32(t *testing.T, values []int64) []uint32 {
	t.Helper()
	converted := make([]uint32, len(values))
	for index, value := range values {
		if value < 0 || value > math.MaxUint32 {
			t.Fatalf("token ID %d is outside uint32", value)
		}
		converted[index] = uint32(value)
	}
	return converted
}

func assertPreparedTensors(t *testing.T, batch preparedBatch, want parity.Tensors) {
	t.Helper()
	if !reflect.DeepEqual(batch.inputIDs, want.InputIDs) {
		t.Fatalf("input_ids do not match corpus")
	}
	if !reflect.DeepEqual(batch.attentionMask, want.AttentionMask) {
		t.Fatalf("attention_mask does not match corpus")
	}
	if !reflect.DeepEqual(batch.markerPos, want.MarkerPos) {
		t.Fatalf("marker_pos does not match corpus")
	}
	if !reflect.DeepEqual(batch.markerMask, want.MarkerMask) {
		t.Fatalf("marker_mask does not match corpus")
	}
	if !reflect.DeepEqual(batch.qtype, want.QType) {
		t.Fatalf("qtype does not match corpus")
	}
}

func assertDerivedRow(
	t *testing.T,
	question preparedQuestion,
	outputs rawModelOutputs,
	row int,
	calibration calibrationConfig,
	want parity.Case,
) {
	t.Helper()
	derived, err := deriveOutputRow(
		question,
		outputs.logits[row][:len(question.options)],
		outputs.actionLogits[row],
		calibration,
	)
	if err != nil {
		t.Fatalf("derive output row: %v", err)
	}
	assertFloatSliceClose(t, "probabilities", derived.probabilities, want.Derived.Probabilities, 1e-12)
	assertFloatSliceClose(t, "action probabilities", derived.actionProbabilities, want.Derived.ActionProbabilities, 1e-12)
	assertFloatClose(t, "confidence", derived.confidence, want.Derived.EntropyConfidence, 1e-12)
	assertFloatClose(t, "temperature", derived.temperature.effective, want.Calibration.EffectiveTemperature, 0)
	rawTemperature, ok := numericTemperature(derived.temperature.raw)
	if !ok {
		t.Fatalf("raw temperature %v is not numeric", derived.temperature.raw)
	}
	assertFloatClose(t, "raw temperature", rawTemperature, want.Calibration.RawTemperature, 0)
	if want.Derived.ExpectedScore != nil {
		assertFloatClose(t, "expected score", derived.expectedScore, *want.Derived.ExpectedScore, 1e-12)
	}
	if want.Derived.TrueProbability != nil {
		assertFloatClose(t, "true probability", derived.trueProbability, *want.Derived.TrueProbability, 1e-12)
	}
}

func assertTypedResult(t *testing.T, result Result, question preparedQuestion, want parity.Case) {
	t.Helper()
	action := roundFour(want.Derived.ActionProbabilities[0])
	switch result := result.(type) {
	case ChoiceResult:
		best := maximumIndex(want.Derived.Probabilities)
		if result.Selected() != CriterionID(question.optionIDs[best]) {
			t.Fatalf("selected choice = %q, want %q", result.Selected(), question.optionIDs[best])
		}
		if result.Confidence() != roundFour(want.Derived.EntropyConfidence) || result.ActionProbability() != action {
			t.Fatalf("choice scalars = (%v,%v), want (%v,%v)", result.Confidence(), result.ActionProbability(), roundFour(want.Derived.EntropyConfidence), action)
		}
		probabilities := result.Probabilities()
		for index, probability := range probabilities {
			if probability.CriterionID != CriterionID(question.optionIDs[index]) || probability.Probability != roundFour(want.Derived.Probabilities[index]) {
				t.Fatalf("choice probability %d = %+v", index, probability)
			}
		}
	case ScoreResult:
		if want.Derived.ExpectedScore == nil {
			t.Fatal("score case has no expected score")
		}
		if result.ExpectedLevel() != roundFour(*want.Derived.ExpectedScore) ||
			result.Confidence() != roundFour(want.Derived.EntropyConfidence) || result.ActionProbability() != action {
			t.Fatalf("score scalars do not match corpus")
		}
		for index, probability := range result.Distribution() {
			if probability.Level != index || probability.Label != question.optionLabels[index] || probability.Probability != roundFour(want.Derived.Probabilities[index]) {
				t.Fatalf("score probability %d = %+v", index, probability)
			}
		}
	case NoulResult:
		if want.Derived.TrueProbability == nil {
			t.Fatal("noul case has no true probability")
		}
		if result.TrueProbability() != roundFour(*want.Derived.TrueProbability) || result.ActionProbability() != action {
			t.Fatalf("noul result = (%v,%v), want (%v,%v)", result.TrueProbability(), result.ActionProbability(), roundFour(*want.Derived.TrueProbability), action)
		}
	default:
		t.Fatalf("unexpected result type %T", result)
	}
}

func float32Rows(values [][]float64) [][]float32 {
	rows := make([][]float32, len(values))
	for row := range values {
		rows[row] = make([]float32, len(values[row]))
		for column, value := range values[row] {
			rows[row][column] = float32(value)
		}
	}
	return rows
}

func cloneRawOutputs(value rawModelOutputs) rawModelOutputs {
	clone := rawModelOutputs{
		logits:       make([][]float32, len(value.logits)),
		actionLogits: make([][]float32, len(value.actionLogits)),
	}
	for row := range value.logits {
		clone.logits[row] = append([]float32(nil), value.logits[row]...)
	}
	for row := range value.actionLogits {
		clone.actionLogits[row] = append([]float32(nil), value.actionLogits[row]...)
	}
	return clone
}

func assertInputTooLong(t *testing.T, err error, inputTokens, limit int) {
	t.Helper()
	if !errors.Is(err, ErrInputTooLong) {
		t.Fatalf("error = %v, want ErrInputTooLong", err)
	}
	var typed *InputTooLongError
	if !errors.As(err, &typed) {
		t.Fatalf("error %T does not contain InputTooLongError", err)
	}
	if typed.InputTokens != inputTokens || typed.Limit != limit {
		t.Fatalf("input-too-long metadata = (%d,%d), want (%d,%d)", typed.InputTokens, typed.Limit, inputTokens, limit)
	}
}

func assertFloatSliceClose(t *testing.T, name string, got, want []float64, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length = %d, want %d", name, len(got), len(want))
	}
	for index := range got {
		assertFloatClose(t, fmt.Sprintf("%s[%d]", name, index), got[index], want[index], tolerance)
	}
}

func assertFloatClose(t *testing.T, name string, got, want, tolerance float64) {
	t.Helper()
	if math.Abs(got-want) > tolerance {
		t.Fatalf("%s = %.17g, want %.17g (tolerance %.3g)", name, got, want, tolerance)
	}
}
