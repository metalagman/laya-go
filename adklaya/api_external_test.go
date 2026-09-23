package adklaya_test

import (
	"errors"
	"testing"

	"github.com/metalagman/laya-go"
	"github.com/metalagman/laya-go/adklaya"
	"google.golang.org/adk/v2/agent"
)

func TestPublicAdapterContractCompiles(t *testing.T) {
	question, err := laya.NewChoiceQuestion("route", "choose", []laya.ChoiceCriterion{
		{ID: "accept", Description: "accept"},
		{ID: "review", Description: "review"},
	})
	if err != nil {
		t.Fatal(err)
	}
	projector := func(_ agent.Context, input string) (laya.State, error) {
		return laya.TextState(input), nil
	}
	_, err = adklaya.NewChoiceNode[string](nil, projector, question, adklaya.NodeOptions{Name: "choice"})
	if !errors.Is(err, laya.ErrInvalidConfig) {
		t.Fatalf("NewChoiceNode(nil model) error = %v, want ErrInvalidConfig", err)
	}

	var _ adklaya.StateProjector[string] = projector
	_ = adklaya.ChoiceOutput{}
	_ = adklaya.ScoreOutput{}
	_ = adklaya.NoulOutput{}
	_ = adklaya.PredictionOutput{}
	_ = adklaya.ChoiceRoutingConfig[string]{}
}
