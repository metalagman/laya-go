package adklaya

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/metalagman/laya-go"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
	"google.golang.org/genai"
)

func TestChoiceNodePreservesPredictionEvidence(t *testing.T) {
	question := mustChoiceQuestion(t)
	prediction := mustChoicePrediction(t, question.ID(), "accept", 0.8, 0.7, 0.1)
	var gotQuestions []laya.Question
	node, err := newChoiceNode(func(_ agent.Context, input string) (laya.State, error) {
		return laya.TextState(input), nil
	}, question, NodeOptions{Name: "choice"}, func(_ agent.Context, _ laya.State, questions []laya.Question) (laya.Prediction, error) {
		gotQuestions = append([]laya.Question(nil), questions...)
		return prediction, nil
	})
	if err != nil {
		t.Fatalf("newChoiceNode() error = %v", err)
	}

	events, err := runNode(t, node)
	if err != nil {
		t.Fatalf("runNode() error = %v", err)
	}
	if len(gotQuestions) != 1 || gotQuestions[0].ID() != question.ID() {
		t.Fatalf("questions = %#v, want one %q question", gotQuestions, question.ID())
	}
	output := findOutput[ChoiceOutput](t, events)
	if output.QuestionID != question.ID() || output.Answer.Selected != "accept" {
		t.Errorf("output = %#v", output)
	}
	if output.Usage != prediction.Usage() || output.Metadata != prediction.Metadata() {
		t.Errorf("evidence = (%#v, %#v), want (%#v, %#v)", output.Usage, output.Metadata, prediction.Usage(), prediction.Metadata())
	}
}

func TestPredictionNodePreservesMixedOrder(t *testing.T) {
	choice := mustChoiceQuestion(t)
	score, err := laya.NewScoreQuestion("severity", "score severity", []laya.ScoreLevel{{Label: "low", Description: "low"}, {Label: "high", Description: "high"}})
	if err != nil {
		t.Fatal(err)
	}
	noul, err := laya.NewNoulQuestion("safe", "is safe", "unsafe", "safe")
	if err != nil {
		t.Fatal(err)
	}
	choiceResult, _ := laya.NewChoiceResult(choice.ID(), laya.ChoiceAnswer{Selected: "accept", Probabilities: []laya.CriterionProbability{{CriterionID: "accept", Probability: 0.8}, {CriterionID: "reject", Probability: 0.2}}, Confidence: 0.7, ActionProbability: 0.1})
	scoreResult, _ := laya.NewScoreResult(score.ID(), laya.ScoreAnswer{ExpectedLevel: 0.8, Distribution: []laya.ScoreProbability{{Level: 0, Label: "low", Probability: 0.2}, {Level: 1, Label: "high", Probability: 0.8}}, Confidence: 0.6, ActionProbability: 0.2})
	noulResult, _ := laya.NewNoulResult(noul.ID(), laya.NoulAnswer{TrueProbability: 0.9, ActionProbability: 0.05})
	prediction, _ := laya.NewPrediction([]laya.Result{choiceResult, scoreResult, noulResult}, laya.Usage{InputTokens: 4, OutputTokens: 3}, laya.PredictionMetadata{ModelID: "fixture"})

	node, err := newPredictionNode(func(_ agent.Context, input string) (laya.State, error) { return laya.TextState(input), nil }, []laya.Question{choice, score, noul}, NodeOptions{Name: "mixed"}, func(_ agent.Context, _ laya.State, _ []laya.Question) (laya.Prediction, error) {
		return prediction, nil
	})
	if err != nil {
		t.Fatalf("newPredictionNode() error = %v", err)
	}
	events, err := runNode(t, node)
	if err != nil {
		t.Fatalf("runNode() error = %v", err)
	}
	output := findOutput[PredictionOutput](t, events)
	want := []ResultKind{ResultKindChoice, ResultKindScore, ResultKindNoul}
	if len(output.Results) != len(want) {
		t.Fatalf("len(results) = %d, want %d", len(output.Results), len(want))
	}
	for i := range want {
		if output.Results[i].Kind != want[i] {
			t.Errorf("results[%d].Kind = %q, want %q", i, output.Results[i].Kind, want[i])
		}
	}
}

func TestScoreAndNoulNodesPreserveAnswers(t *testing.T) {
	projector := func(_ agent.Context, input string) (laya.State, error) { return laya.TextState(input), nil }
	t.Run("score", func(t *testing.T) {
		question, err := laya.NewScoreQuestion("severity", "score", []laya.ScoreLevel{{Label: "low", Description: "low"}, {Label: "high", Description: "high"}})
		if err != nil {
			t.Fatal(err)
		}
		result, _ := laya.NewScoreResult(question.ID(), laya.ScoreAnswer{ExpectedLevel: 0.75, Distribution: []laya.ScoreProbability{{Level: 0, Label: "low", Probability: 0.25}, {Level: 1, Label: "high", Probability: 0.75}}, Confidence: 0.8, ActionProbability: 0.1})
		prediction, _ := laya.NewPrediction([]laya.Result{result}, laya.Usage{InputTokens: 2, OutputTokens: 1}, laya.PredictionMetadata{ModelID: "fixture"})
		node, err := newScoreNode(projector, question, NodeOptions{Name: "score"}, func(agent.Context, laya.State, []laya.Question) (laya.Prediction, error) { return prediction, nil })
		if err != nil {
			t.Fatal(err)
		}
		events, err := runNode(t, node)
		if err != nil {
			t.Fatal(err)
		}
		output := findOutput[ScoreOutput](t, events)
		if output.Answer.ExpectedLevel != 0.75 || len(output.Answer.Distribution) != 2 || output.Usage != prediction.Usage() {
			t.Errorf("output = %#v", output)
		}
	})
	t.Run("noul", func(t *testing.T) {
		question, err := laya.NewNoulQuestion("safe", "safe", "no", "yes")
		if err != nil {
			t.Fatal(err)
		}
		result, _ := laya.NewNoulResult(question.ID(), laya.NoulAnswer{TrueProbability: 0.9, ActionProbability: 0.05})
		prediction, _ := laya.NewPrediction([]laya.Result{result}, laya.Usage{InputTokens: 2, OutputTokens: 1}, laya.PredictionMetadata{ModelID: "fixture"})
		node, err := newNoulNode(projector, question, NodeOptions{Name: "noul"}, func(agent.Context, laya.State, []laya.Question) (laya.Prediction, error) { return prediction, nil })
		if err != nil {
			t.Fatal(err)
		}
		events, err := runNode(t, node)
		if err != nil {
			t.Fatal(err)
		}
		output := findOutput[NoulOutput](t, events)
		if output.Answer.TrueProbability != 0.9 || output.Answer.ActionProbability != 0.05 || output.Metadata != prediction.Metadata() {
			t.Errorf("output = %#v", output)
		}
	})
}

func TestOutputContractsSupportJSONAndSchemaGeneration(t *testing.T) {
	types := []struct {
		name  string
		value any
	}{
		{name: "choice", value: ChoiceOutput{QuestionID: "q", Answer: laya.ChoiceAnswer{Selected: "a"}}},
		{name: "score", value: ScoreOutput{QuestionID: "q", Answer: laya.ScoreAnswer{ExpectedLevel: 1}}},
		{name: "noul", value: NoulOutput{QuestionID: "q", Answer: laya.NoulAnswer{TrueProbability: 0.5}}},
		{name: "mixed", value: PredictionOutput{Results: []ResultOutput{{Kind: ResultKindChoice, QuestionID: "q", Choice: &laya.ChoiceAnswer{Selected: "a"}}}}},
	}
	for _, tc := range types {
		t.Run(tc.name, func(t *testing.T) {
			data, err := json.Marshal(tc.value)
			if err != nil {
				t.Fatalf("json.Marshal() error = %v", err)
			}
			if !json.Valid(data) {
				t.Fatalf("json.Marshal() returned invalid JSON %q", data)
			}
		})
	}
	for name, makeSchema := range map[string]func() (*jsonschema.Schema, error){
		"choice":     func() (*jsonschema.Schema, error) { return jsonschema.For[ChoiceOutput](nil) },
		"score":      func() (*jsonschema.Schema, error) { return jsonschema.For[ScoreOutput](nil) },
		"noul":       func() (*jsonschema.Schema, error) { return jsonschema.For[NoulOutput](nil) },
		"prediction": func() (*jsonschema.Schema, error) { return jsonschema.For[PredictionOutput](nil) },
	} {
		t.Run(name+" schema", func(t *testing.T) {
			schema, err := makeSchema()
			if err != nil {
				t.Fatalf("jsonschema.For() error = %v", err)
			}
			if _, err := schema.Resolve(nil); err != nil {
				t.Fatalf("schema.Resolve() error = %v", err)
			}
		})
	}
}

func TestDataNodeReturnsProjectorAndPredictionErrors(t *testing.T) {
	errProject := errors.New("project failed")
	errPredict := errors.New("predict failed")
	question := mustChoiceQuestion(t)
	tests := []struct {
		name      string
		projector StateProjector[string]
		predict   predictFunc
		want      error
	}{
		{
			name: "projector",
			projector: func(agent.Context, string) (laya.State, error) {
				return laya.State{}, errProject
			},
			predict: func(agent.Context, laya.State, []laya.Question) (laya.Prediction, error) {
				t.Fatal("predict called after projector error")
				return laya.Prediction{}, nil
			},
			want: errProject,
		},
		{
			name: "prediction",
			projector: func(agent.Context, string) (laya.State, error) {
				return laya.TextState("state"), nil
			},
			predict: func(agent.Context, laya.State, []laya.Question) (laya.Prediction, error) {
				return laya.Prediction{}, errPredict
			},
			want: errPredict,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			node, err := newChoiceNode(tc.projector, question, NodeOptions{Name: "choice"}, tc.predict)
			if err != nil {
				t.Fatal(err)
			}
			_, err = runNode(t, node)
			if !errors.Is(err, tc.want) {
				t.Fatalf("run error = %v, want errors.Is(_, %v)", err, tc.want)
			}
		})
	}
}

func TestNodeOptionsAreValidatedAndCopied(t *testing.T) {
	question := mustChoiceQuestion(t)
	projector := func(agent.Context, string) (laya.State, error) { return laya.TextState("state"), nil }
	predict := func(agent.Context, laya.State, []laya.Question) (laya.Prediction, error) {
		return mustChoicePrediction(t, question.ID(), "accept", 0.8, 0.7, 0.1), nil
	}
	for _, name := range []string{"", "   ", "user"} {
		if _, err := newChoiceNode(projector, question, NodeOptions{Name: name}, predict); err == nil {
			t.Errorf("newChoiceNode(name %q) succeeded, want error", name)
		}
	}
	rerun := true
	retry := workflow.DefaultRetryConfig()
	node, err := newChoiceNode(projector, question, NodeOptions{Name: "choice", Config: workflow.NodeConfig{RerunOnResume: &rerun, RetryConfig: retry}}, predict)
	if err != nil {
		t.Fatal(err)
	}
	rerun = false
	retry.MaxAttempts = 99
	if !*node.Config().RerunOnResume || node.Config().RetryConfig.MaxAttempts == 99 {
		t.Errorf("node config retained caller pointers: %#v", node.Config())
	}
}

func runNode(t *testing.T, node workflow.Node) ([]*session.Event, error) {
	t.Helper()
	ingress := workflow.NewFunctionNode("ingress", func(_ agent.Context, raw any) (string, error) {
		if input, ok := raw.(string); ok {
			return input, nil
		}
		var input *genai.Content
		switch value := raw.(type) {
		case *genai.Content:
			input = value
		case genai.Content:
			input = &value
		}
		if input == nil || len(input.Parts) == 0 {
			return "", fmt.Errorf("missing input")
		}
		return input.Parts[0].Text, nil
	}, workflow.NodeConfig{})
	a, err := workflowagent.New(workflowagent.Config{Name: "test_workflow", Edges: workflow.Chain(workflow.Start, ingress, node)})
	if err != nil {
		t.Fatalf("workflowagent.New() error = %v", err)
	}
	service := session.InMemoryService()
	r, err := runner.New(runner.Config{AppName: "adklaya_test", Agent: a, SessionService: service})
	if err != nil {
		t.Fatalf("runner.NewInMemory() error = %v", err)
	}
	ctx := context.Background()
	if _, err := service.Create(ctx, &session.CreateRequest{AppName: "adklaya_test", UserID: "test", SessionID: "session"}); err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	var events []*session.Event
	for event, runErr := range r.Run(ctx, "test", "session", genai.NewContentFromText("state", genai.RoleUser), agent.RunConfig{}) {
		if runErr != nil {
			return events, runErr
		}
		events = append(events, event)
	}
	return events, nil
}

func findOutput[T any](t *testing.T, events []*session.Event) T {
	t.Helper()
	for _, event := range events {
		if output, ok := event.Output.(T); ok {
			return output
		}
	}
	var zero T
	t.Fatalf("events contain no %T output", zero)
	return zero
}

func mustChoiceQuestion(t *testing.T) laya.ChoiceQuestion {
	t.Helper()
	question, err := laya.NewChoiceQuestion("decision", "choose", []laya.ChoiceCriterion{{ID: "accept", Description: "accept"}, {ID: "reject", Description: "reject"}})
	if err != nil {
		t.Fatal(err)
	}
	return question
}

func mustChoicePrediction(t *testing.T, questionID laya.QuestionID, selected laya.CriterionID, selectedProbability, confidence, action float64) laya.Prediction {
	t.Helper()
	other := laya.CriterionID("reject")
	if selected == other {
		other = "accept"
	}
	result, err := laya.NewChoiceResult(questionID, laya.ChoiceAnswer{Selected: selected, Probabilities: []laya.CriterionProbability{{CriterionID: selected, Probability: selectedProbability}, {CriterionID: other, Probability: 1 - selectedProbability}}, Confidence: confidence, ActionProbability: action})
	if err != nil {
		t.Fatal(err)
	}
	prediction, err := laya.NewPrediction([]laya.Result{result}, laya.Usage{InputTokens: 3, OutputTokens: 1}, laya.PredictionMetadata{ModelID: "fixture", BundleID: "bundle"})
	if err != nil {
		t.Fatal(err)
	}
	return prediction
}
