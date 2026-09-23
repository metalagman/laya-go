package laya_test

import (
	"context"
	"io/fs"
	"testing"

	laya "github.com/metalagman/laya-go"
)

var (
	_ error                                                                                          = laya.ErrMaterialization
	_ func(context.Context, laya.RuntimeOptions) (*laya.Runtime, error)                              = laya.NewRuntime
	_ func(*laya.Runtime, context.Context, string, laya.ModelOptions) (*laya.Model, error)           = (*laya.Runtime).OpenModelDir
	_ func(*laya.Runtime, context.Context, fs.FS, laya.FSModelOptions) (*laya.Model, error)          = (*laya.Runtime).OpenModelFS
	_ func(*laya.Model, context.Context, laya.State, []laya.Question) (laya.Prediction, error)       = (*laya.Model).Predict
	_ func(*laya.Model, context.Context, laya.State, laya.ChoiceQuestion) (laya.ChoiceResult, error) = (*laya.Model).Choice
	_ func(*laya.Model, context.Context, laya.State, laya.ScoreQuestion) (laya.ScoreResult, error)   = (*laya.Model).Score
	_ func(*laya.Model, context.Context, laya.State, laya.NoulQuestion) (laya.NoulResult, error)     = (*laya.Model).Noul
	_ func(*laya.Model, context.Context) error                                                       = (*laya.Model).Close
	_ func(*laya.Runtime, context.Context) error                                                     = (*laya.Runtime).Close
)

func TestPublicValueContracts(t *testing.T) {
	state, err := laya.JSONState([]byte(`{"customer":{"age":42}}`))
	if err != nil {
		t.Fatalf("laya.JSONState() returned unexpected error: %v", err)
	}
	if state.Kind() != laya.StateKindJSON {
		t.Errorf("state.Kind() = %v, want %v", state.Kind(), laya.StateKindJSON)
	}

	choiceQuestion, err := laya.NewChoiceQuestion("route", "Choose a route.", []laya.ChoiceCriterion{
		{ID: "safe", Description: "proceed"},
		{ID: "review", Description: "request review"},
	})
	if err != nil {
		t.Fatalf("laya.NewChoiceQuestion() returned unexpected error: %v", err)
	}
	scoreQuestion, err := laya.NewScoreQuestion("quality", "Score quality.", []laya.ScoreLevel{
		{Label: "low", Description: "low quality"},
		{Label: "high", Description: "high quality"},
	})
	if err != nil {
		t.Fatalf("laya.NewScoreQuestion() returned unexpected error: %v", err)
	}
	noulQuestion, err := laya.NewNoulQuestion("valid", "Is this valid?", "invalid", "valid")
	if err != nil {
		t.Fatalf("laya.NewNoulQuestion() returned unexpected error: %v", err)
	}
	questions := []laya.Question{choiceQuestion, scoreQuestion, noulQuestion}
	if got, want := len(questions), 3; got != want {
		t.Fatalf("len(questions) = %d, want %d", got, want)
	}

	choiceResult, err := laya.NewChoiceResult("route", laya.ChoiceAnswer{
		Selected: "safe",
		Probabilities: []laya.CriterionProbability{
			{CriterionID: "safe", Probability: 0.8},
			{CriterionID: "review", Probability: 0.2},
		},
		Confidence:        0.9,
		ActionProbability: 0.1,
	})
	if err != nil {
		t.Fatalf("laya.NewChoiceResult() returned unexpected error: %v", err)
	}
	scoreResult, err := laya.NewScoreResult("quality", laya.ScoreAnswer{
		ExpectedLevel: 0.7,
		Distribution: []laya.ScoreProbability{
			{Level: 0, Label: "low", Probability: 0.3},
			{Level: 1, Label: "high", Probability: 0.7},
		},
		Confidence:        0.8,
		ActionProbability: 0.2,
	})
	if err != nil {
		t.Fatalf("laya.NewScoreResult() returned unexpected error: %v", err)
	}
	noulResult, err := laya.NewNoulResult("valid", laya.NoulAnswer{
		TrueProbability:   0.95,
		ActionProbability: 0.05,
	})
	if err != nil {
		t.Fatalf("laya.NewNoulResult() returned unexpected error: %v", err)
	}
	prediction, err := laya.NewPrediction(
		[]laya.Result{choiceResult, scoreResult, noulResult},
		laya.Usage{InputTokens: 20, OutputTokens: 3},
		laya.PredictionMetadata{
			ModelID:   "laya",
			BundleID:  "sha256:example",
			RuntimeID: "runtime-1",
			Truncation: laya.TruncationMetadata{
				OriginalInputTokens:  20,
				EffectiveInputTokens: 20,
			},
		},
	)
	if err != nil {
		t.Fatalf("laya.NewPrediction() returned unexpected error: %v", err)
	}
	if got, want := len(prediction.Results()), 3; got != want {
		t.Errorf("len(prediction.Results()) = %d, want %d", got, want)
	}
	if got, want := prediction.Usage().InputTokens, 20; got != want {
		t.Errorf("prediction.Usage().InputTokens = %d, want %d", got, want)
	}
	if got, want := choiceResult.Selected(), laya.CriterionID("safe"); got != want {
		t.Errorf("choiceResult.Selected() = %q, want %q", got, want)
	}
	if got, want := scoreResult.ExpectedLevel(), 0.7; got != want {
		t.Errorf("scoreResult.ExpectedLevel() = %v, want %v", got, want)
	}
	if got, want := noulResult.TrueProbability(), 0.95; got != want {
		t.Errorf("noulResult.TrueProbability() = %v, want %v", got, want)
	}

	_ = laya.RuntimeOptions{}
	_ = laya.ModelOptions{QueueCapacity: 1, InputTokenLimit: 512, Truncation: laya.RejectOverflow}
	_ = laya.FSModelOptions{Root: ".", WorkDir: "/tmp/laya", ModelOptions: laya.ModelOptions{}}
}
