package laya_test

import (
	"context"
	"errors"
	"fmt"
	"os"

	laya "github.com/metalagman/laya-go"
)

func ExampleTextState() {
	state := laya.TextState("  exact text\n")
	text, ok := state.Text()
	fmt.Printf("%t %q\n", ok, text)

	// Output:
	// true "  exact text\n"
}

func ExampleJSONState() {
	state, err := laya.JSONState([]byte(` { "b": 2, "a": [1, true] } `))
	if err != nil {
		fmt.Println(err)
		return
	}
	data, ok := state.JSON()
	fmt.Printf("%t %s\n", ok, data)

	// Output:
	// true {"b":2,"a":[1,true]}
}

func ExampleNewPrediction() {
	choice, err := laya.NewChoiceResult("route", laya.ChoiceAnswer{
		Selected: "accept",
		Probabilities: []laya.CriterionProbability{
			{CriterionID: "accept", Probability: 0.8},
			{CriterionID: "review", Probability: 0.2},
		},
		Confidence:        0.9,
		ActionProbability: 0.1,
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	score, err := laya.NewScoreResult("quality", laya.ScoreAnswer{
		ExpectedLevel: 0.75,
		Distribution: []laya.ScoreProbability{
			{Level: 0, Label: "low", Probability: 0.25},
			{Level: 1, Label: "high", Probability: 0.75},
		},
		Confidence:        0.8,
		ActionProbability: 0.2,
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	noul, err := laya.NewNoulResult("valid", laya.NoulAnswer{
		TrueProbability:   0.95,
		ActionProbability: 0.05,
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	prediction, err := laya.NewPrediction(
		[]laya.Result{choice, score, noul},
		laya.Usage{InputTokens: 12, OutputTokens: 3},
		laya.PredictionMetadata{},
	)
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Printf("results=%d input=%d output=%d\n", len(prediction.Results()), prediction.Usage().InputTokens, prediction.Usage().OutputTokens)
	fmt.Printf("choice=%s confidence=%.1f action=%.1f\n", choice.Selected(), choice.Confidence(), choice.ActionProbability())
	fmt.Printf("score=%.2f confidence=%.1f action=%.1f\n", score.ExpectedLevel(), score.Confidence(), score.ActionProbability())
	fmt.Printf("true=%.2f action=%.2f\n", noul.TrueProbability(), noul.ActionProbability())

	// Output:
	// results=3 input=12 output=3
	// choice=accept confidence=0.9 action=0.1
	// score=0.75 confidence=0.8 action=0.2
	// true=0.95 action=0.05
}

func ExampleNewRuntime() {
	ctx := context.Background()
	runtime, err := laya.NewRuntime(ctx, laya.RuntimeOptions{})
	if errors.Is(err, laya.ErrNativeUnavailable) {
		fmt.Println("native runtime unavailable")
		return
	}
	if err != nil {
		fmt.Println(err)
		return
	}
	defer func() {
		if err := runtime.Close(context.Background()); err != nil {
			fmt.Println(err)
		}
	}()

	model, err := runtime.OpenModelDir(ctx, "/caller-owned/model-snapshot", laya.ModelOptions{})
	if err != nil {
		fmt.Println(err)
		return
	}
	defer func() {
		if err := model.Close(context.Background()); err != nil {
			fmt.Println(err)
		}
	}()

	// Output:
	// native runtime unavailable
}

func ExampleRuntime_OpenModelFS() {
	ctx := context.Background()
	runtime, err := laya.NewRuntime(ctx, laya.RuntimeOptions{})
	if errors.Is(err, laya.ErrNativeUnavailable) {
		fmt.Println("native runtime unavailable")
		return
	}
	if err != nil {
		fmt.Println(err)
		return
	}
	defer func() {
		if err := runtime.Close(context.Background()); err != nil {
			fmt.Println(err)
		}
	}()

	// An embed.FS can be supplied in the same position. The caller owns the
	// complete bundle filesystem and the existing materialization work directory.
	bundle := os.DirFS("/caller-owned/complete-bundle")
	model, err := runtime.OpenModelFS(ctx, bundle, laya.FSModelOptions{
		Root:    ".",
		WorkDir: "/caller-owned/materialization",
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	defer func() {
		if err := model.Close(context.Background()); err != nil {
			fmt.Println(err)
		}
	}()

	// Output:
	// native runtime unavailable
}
