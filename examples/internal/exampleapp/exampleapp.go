// Package exampleapp contains shared application-level example contracts.
package exampleapp

import (
	"context"
	"encoding/json"
	"fmt"
	"io"

	"github.com/metalagman/laya-go"
)

// Report is a content-safe summary of one example inference batch.
type Report struct {
	Choice    laya.CriterionID `json:"choice"`
	Score     float64          `json:"score"`
	Noul      float64          `json:"noul"`
	BundleID  laya.BundleID    `json:"bundleId"`
	RuntimeID laya.RuntimeID   `json:"runtimeId"`
}

// Questions returns the three immutable questions used by the examples.
func Questions() ([]laya.Question, error) {
	choice, err := laya.NewChoiceQuestion("route", "Choose a route", []laya.ChoiceCriterion{
		{ID: "accept", Description: "continue automatically"},
		{ID: "review", Description: "request human review"},
	})
	if err != nil {
		return nil, err
	}
	score, err := laya.NewScoreQuestion("urgency", "Score urgency", []laya.ScoreLevel{
		{Label: "low", Description: "not urgent"},
		{Label: "high", Description: "urgent"},
	})
	if err != nil {
		return nil, err
	}
	noul, err := laya.NewNoulQuestion("safe", "Is the request safe?", "unsafe", "safe")
	if err != nil {
		return nil, err
	}
	return []laya.Question{choice, score, noul}, nil
}

// Evaluate runs all primitives as one ordered batch and writes no raw input.
func Evaluate(ctx context.Context, model *laya.Model, state laya.State, output io.Writer) error {
	if model == nil {
		return fmt.Errorf("evaluate example: %w: model is nil", laya.ErrInvalidConfig)
	}
	if output == nil {
		return fmt.Errorf("evaluate example: %w: output is nil", laya.ErrInvalidConfig)
	}
	questions, err := Questions()
	if err != nil {
		return fmt.Errorf("create example questions: %w", err)
	}
	prediction, err := model.Predict(ctx, state, questions)
	if err != nil {
		return fmt.Errorf("predict example batch: %w", err)
	}
	results := prediction.Results()
	choice, ok := results[0].(laya.ChoiceResult)
	if !ok {
		return fmt.Errorf("example choice: %w", laya.ErrInvalidOutput)
	}
	score, ok := results[1].(laya.ScoreResult)
	if !ok {
		return fmt.Errorf("example score: %w", laya.ErrInvalidOutput)
	}
	noul, ok := results[2].(laya.NoulResult)
	if !ok {
		return fmt.Errorf("example noul: %w", laya.ErrInvalidOutput)
	}
	report := Report{Choice: choice.Selected(), Score: score.ExpectedLevel(), Noul: noul.TrueProbability(), BundleID: prediction.Metadata().BundleID, RuntimeID: prediction.Metadata().RuntimeID}
	if err := json.NewEncoder(output).Encode(report); err != nil {
		return fmt.Errorf("encode content-safe example report: %w", err)
	}
	return nil
}
