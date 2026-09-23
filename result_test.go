package laya

import (
	"errors"
	"math"
	"testing"
)

func newChoiceResult(questionID QuestionID, selected CriterionID, probabilities []CriterionProbability, confidence, actionProbability float64) (ChoiceResult, error) {
	return NewChoiceResult(questionID, ChoiceAnswer{
		Selected:          selected,
		Probabilities:     probabilities,
		Confidence:        confidence,
		ActionProbability: actionProbability,
	})
}

func newScoreResult(questionID QuestionID, expectedLevel float64, distribution []ScoreProbability, confidence, actionProbability float64) (ScoreResult, error) {
	return NewScoreResult(questionID, ScoreAnswer{
		ExpectedLevel:     expectedLevel,
		Distribution:      distribution,
		Confidence:        confidence,
		ActionProbability: actionProbability,
	})
}

func newNoulResult(questionID QuestionID, trueProbability, actionProbability float64) (NoulResult, error) {
	return NewNoulResult(questionID, NoulAnswer{
		TrueProbability:   trueProbability,
		ActionProbability: actionProbability,
	})
}

func TestChoiceResultCopiesProbabilities(t *testing.T) {
	probabilities := []CriterionProbability{
		{CriterionID: "safe", Probability: 0.75},
		{CriterionID: "review", Probability: 0.25},
	}
	result, err := newChoiceResult("route", "safe", probabilities, 0.9, 0.1)
	if err != nil {
		t.Fatalf("NewChoiceResult() returned unexpected error: %v", err)
	}
	probabilities[0].Probability = 0

	got := result.Probabilities()
	if got[0].Probability != 0.75 {
		t.Errorf("result.Probabilities()[0].Probability = %v, want %v", got[0].Probability, 0.75)
	}
	got[0].Probability = 0
	if gotAgain := result.Probabilities()[0].Probability; gotAgain != 0.75 {
		t.Errorf("result.Probabilities()[0].Probability after caller mutation = %v, want %v", gotAgain, 0.75)
	}
	if got, want := result.Selected(), CriterionID("safe"); got != want {
		t.Errorf("result.Selected() = %q, want %q", got, want)
	}
	if got, want := result.Confidence(), 0.9; got != want {
		t.Errorf("result.Confidence() = %v, want %v", got, want)
	}
	if got, want := result.ActionProbability(), 0.1; got != want {
		t.Errorf("result.ActionProbability() = %v, want %v", got, want)
	}
}

func TestScoreResultContract(t *testing.T) {
	distribution := []ScoreProbability{
		{Level: 0, Label: "low", Probability: 0.25},
		{Level: 1, Label: "high", Probability: 0.75},
	}
	result, err := newScoreResult("quality", 0.75, distribution, 0.8, 0.2)
	if err != nil {
		t.Fatalf("NewScoreResult() returned unexpected error: %v", err)
	}
	distribution[0].Label = "mutated"

	if got, want := result.ExpectedLevel(), 0.75; got != want {
		t.Errorf("result.ExpectedLevel() = %v, want %v", got, want)
	}
	if got, want := result.Distribution()[0].Label, "low"; got != want {
		t.Errorf("result.Distribution()[0].Label = %q, want %q", got, want)
	}
	gotDistribution := result.Distribution()
	gotDistribution[0].Label = "also-mutated"
	if got, want := result.Distribution()[0].Label, "low"; got != want {
		t.Errorf("result.Distribution()[0].Label after caller mutation = %q, want %q", got, want)
	}
	if got, want := result.Confidence(), 0.8; got != want {
		t.Errorf("result.Confidence() = %v, want %v", got, want)
	}
	if got, want := result.ActionProbability(), 0.2; got != want {
		t.Errorf("result.ActionProbability() = %v, want %v", got, want)
	}
}

func TestNoulResultContract(t *testing.T) {
	result, err := newNoulResult("valid", 0.85, 0.05)
	if err != nil {
		t.Fatalf("NewNoulResult() returned unexpected error: %v", err)
	}
	if got, want := result.TrueProbability(), 0.85; got != want {
		t.Errorf("result.TrueProbability() = %v, want %v", got, want)
	}
	if got, want := result.ActionProbability(), 0.05; got != want {
		t.Errorf("result.ActionProbability() = %v, want %v", got, want)
	}
}

func TestPredictionOwnsOrderedResultsAndSharedUsage(t *testing.T) {
	choice, err := newChoiceResult("route", "safe", []CriterionProbability{{CriterionID: "safe", Probability: 0.8}, {CriterionID: "review", Probability: 0.2}}, 0.9, 0.1)
	if err != nil {
		t.Fatalf("NewChoiceResult() returned unexpected error: %v", err)
	}
	noul, err := newNoulResult("valid", 0.7, 0.2)
	if err != nil {
		t.Fatalf("NewNoulResult() returned unexpected error: %v", err)
	}
	results := []Result{choice, noul}
	metadata := PredictionMetadata{
		ModelID:   "laya",
		BundleID:  "bundle-sha256",
		RuntimeID: "runtime-1",
		Truncation: TruncationMetadata{
			OriginalInputTokens:  12,
			EffectiveInputTokens: 12,
		},
	}
	prediction, err := NewPrediction(results, Usage{InputTokens: 12, OutputTokens: 2}, metadata)
	if err != nil {
		t.Fatalf("NewPrediction() returned unexpected error: %v", err)
	}
	results[0] = noul

	got := prediction.Results()
	if got[0].QuestionID() != "route" || got[1].QuestionID() != "valid" {
		t.Errorf("prediction.Results() IDs = [%q, %q], want [route, valid]", got[0].QuestionID(), got[1].QuestionID())
	}
	got[0] = noul
	if gotAgain := prediction.Results()[0].QuestionID(); gotAgain != "route" {
		t.Errorf("prediction.Results()[0].QuestionID() after caller mutation = %q, want %q", gotAgain, QuestionID("route"))
	}
	if got, want := prediction.Usage(), (Usage{InputTokens: 12, OutputTokens: 2}); got != want {
		t.Errorf("prediction.Usage() = %+v, want %+v", got, want)
	}
	if got, want := prediction.Metadata(), metadata; got != want {
		t.Errorf("prediction.Metadata() = %+v, want %+v", got, want)
	}
}

func TestResultValidation(t *testing.T) {
	tests := []struct {
		name string
		call func() error
	}{
		{
			name: "choice empty question ID",
			call: func() error {
				_, err := newChoiceResult("", "safe", []CriterionProbability{{CriterionID: "safe", Probability: 1}}, 1, 0)
				return err
			},
		},
		{
			name: "choice empty selected",
			call: func() error {
				_, err := newChoiceResult("route", "", []CriterionProbability{{CriterionID: "safe", Probability: 1}}, 1, 0)
				return err
			},
		},
		{
			name: "choice empty probabilities",
			call: func() error {
				_, err := newChoiceResult("route", "safe", nil, 1, 0)
				return err
			},
		},
		{
			name: "choice empty probability ID",
			call: func() error {
				_, err := newChoiceResult("route", "safe", []CriterionProbability{{Probability: 1}}, 1, 0)
				return err
			},
		},
		{
			name: "choice duplicate probability",
			call: func() error {
				_, err := newChoiceResult("route", "safe", []CriterionProbability{{CriterionID: "safe", Probability: 0.5}, {CriterionID: "safe", Probability: 0.5}}, 1, 0)
				return err
			},
		},
		{
			name: "choice selected absent",
			call: func() error {
				_, err := newChoiceResult("route", "missing", []CriterionProbability{{CriterionID: "safe", Probability: 1}}, 1, 0)
				return err
			},
		},
		{
			name: "choice probability not finite",
			call: func() error {
				_, err := newChoiceResult("route", "safe", []CriterionProbability{{CriterionID: "safe", Probability: math.NaN()}}, 1, 0)
				return err
			},
		},
		{
			name: "choice confidence out of range",
			call: func() error {
				_, err := newChoiceResult("route", "safe", []CriterionProbability{{CriterionID: "safe", Probability: 1}}, -0.1, 0)
				return err
			},
		},
		{
			name: "choice action out of range",
			call: func() error {
				_, err := newChoiceResult("route", "safe", []CriterionProbability{{CriterionID: "safe", Probability: 1}}, 1, 1.1)
				return err
			},
		},
		{
			name: "score empty question ID",
			call: func() error {
				_, err := newScoreResult("", 0, []ScoreProbability{{Level: 0, Label: "one", Probability: 1}}, 1, 0)
				return err
			},
		},
		{
			name: "score empty distribution",
			call: func() error {
				_, err := newScoreResult("score", 0, nil, 1, 0)
				return err
			},
		},
		{
			name: "score expected out of range",
			call: func() error {
				_, err := newScoreResult("score", 1, []ScoreProbability{{Level: 0, Label: "one", Probability: 1}}, 1, 0)
				return err
			},
		},
		{
			name: "score levels out of order",
			call: func() error {
				_, err := newScoreResult("score", 0, []ScoreProbability{{Level: 1, Label: "wrong", Probability: 1}}, 1, 0)
				return err
			},
		},
		{
			name: "score empty label",
			call: func() error {
				_, err := newScoreResult("score", 0, []ScoreProbability{{Level: 0, Probability: 1}}, 1, 0)
				return err
			},
		},
		{
			name: "score probability out of range",
			call: func() error {
				_, err := newScoreResult("score", 0, []ScoreProbability{{Level: 0, Label: "one", Probability: -0.1}}, 1, 0)
				return err
			},
		},
		{
			name: "score confidence out of range",
			call: func() error {
				_, err := newScoreResult("score", 0, []ScoreProbability{{Level: 0, Label: "one", Probability: 1}}, 1.1, 0)
				return err
			},
		},
		{
			name: "score action out of range",
			call: func() error {
				_, err := newScoreResult("score", 0, []ScoreProbability{{Level: 0, Label: "one", Probability: 1}}, 1, -0.1)
				return err
			},
		},
		{
			name: "noul empty question ID",
			call: func() error {
				_, err := newNoulResult("", 0.5, 0)
				return err
			},
		},
		{
			name: "noul probability out of range",
			call: func() error {
				_, err := newNoulResult("valid", 1.1, 0)
				return err
			},
		},
		{
			name: "noul action out of range",
			call: func() error {
				_, err := newNoulResult("valid", 0.5, math.Inf(1))
				return err
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if err := test.call(); !errors.Is(err, ErrInvalidOutput) {
				t.Errorf("result constructor error = %v, want errors.Is(_, ErrInvalidOutput)", err)
			}
		})
	}
}

func TestPredictionValidation(t *testing.T) {
	result, err := newNoulResult("valid", 0.5, 0.25)
	if err != nil {
		t.Fatalf("NewNoulResult() returned unexpected error: %v", err)
	}
	tests := []struct {
		name       string
		results    []Result
		usage      Usage
		truncation TruncationMetadata
	}{
		{name: "no results"},
		{name: "duplicate question", results: []Result{result, result}},
		{name: "negative usage", results: []Result{result}, usage: Usage{InputTokens: -1}},
		{name: "inconsistent truncation", results: []Result{result}, truncation: TruncationMetadata{Truncated: true, OriginalInputTokens: 10, EffectiveInputTokens: 10}},
		{name: "negative truncation count", results: []Result{result}, truncation: TruncationMetadata{OriginalInputTokens: -1, EffectiveInputTokens: -1}},
		{name: "untruncated counts differ", results: []Result{result}, truncation: TruncationMetadata{OriginalInputTokens: 10, EffectiveInputTokens: 9}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := NewPrediction(test.results, test.usage, PredictionMetadata{Truncation: test.truncation})
			if !errors.Is(err, ErrInvalidOutput) {
				t.Errorf("NewPrediction() error = %v, want errors.Is(_, ErrInvalidOutput)", err)
			}
		})
	}
}

func TestValidateResultVariants(t *testing.T) {
	choice, err := newChoiceResult("route", "safe", []CriterionProbability{{CriterionID: "safe", Probability: 1}}, 1, 0)
	if err != nil {
		t.Fatalf("NewChoiceResult() returned unexpected error: %v", err)
	}
	score, err := newScoreResult("score", 0, []ScoreProbability{{Level: 0, Label: "one", Probability: 1}}, 1, 0)
	if err != nil {
		t.Fatalf("NewScoreResult() returned unexpected error: %v", err)
	}
	noul, err := newNoulResult("valid", 0.5, 0)
	if err != nil {
		t.Fatalf("NewNoulResult() returned unexpected error: %v", err)
	}

	valid := []Result{choice, &choice, score, &score, noul, &noul}
	for _, result := range valid {
		if err := validateResult(result); err != nil {
			t.Errorf("validateResult(%T) = %v, want nil", result, err)
		}
	}

	var nilChoice *ChoiceResult
	var nilScore *ScoreResult
	var nilNoul *NoulResult
	invalid := []Result{nilChoice, nilScore, nilNoul}
	for _, result := range invalid {
		if err := validateResult(result); !errors.Is(err, ErrInvalidOutput) {
			t.Errorf("validateResult(%T(nil)) = %v, want errors.Is(_, ErrInvalidOutput)", result, err)
		}
	}
}
