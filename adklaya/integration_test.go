package adklaya

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/metalagman/laya-go"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
	"google.golang.org/genai"
)

func TestRunnerRetriesWholeAdapterAttempt(t *testing.T) {
	question := mustChoiceQuestion(t)
	prediction := mustChoicePrediction(t, question.ID(), "accept", 0.8, 0.7, 0.1)
	var projections atomic.Int32
	var predictions atomic.Int32
	retry := workflow.DefaultRetryConfig()
	retry.MaxAttempts = 3
	retry.InitialDelay = 0
	retry.MaxDelay = 0
	node, err := newChoiceNode(func(_ agent.Context, input string) (laya.State, error) {
		projections.Add(1)
		return laya.TextState(input), nil
	}, question, NodeOptions{Name: "choice", Config: workflow.NodeConfig{RetryConfig: retry}}, func(_ agent.Context, _ laya.State, _ []laya.Question) (laya.Prediction, error) {
		if predictions.Add(1) < 3 {
			return laya.Prediction{}, errors.New("transient")
		}
		return prediction, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	events, err := runNode(t, node)
	if err != nil {
		t.Fatalf("runNode() error = %v", err)
	}
	_ = findOutput[ChoiceOutput](t, events)
	if projections.Load() != 3 || predictions.Load() != 3 {
		t.Errorf("projections = %d, predictions = %d; want 3 whole attempts", projections.Load(), predictions.Load())
	}
}

func TestRoutingTimeoutEmitsNoLateRoute(t *testing.T) {
	accept := terminalNode("accepted", "accepted")
	fallback := terminalNode("fallback", "fallback")
	policy := &AcceptancePolicy{MaxActionProbability: 1}
	routing, err := newChoiceRouting(ChoiceRoutingConfig[string]{
		NodeOptions: NodeOptions{Name: "route", Config: workflow.NodeConfig{Timeout: 20 * time.Millisecond}},
		QuestionID:  "decision", Instructions: "choose", ProjectState: textProjector,
		Branches: []ChoiceBranch{{Criteria: []laya.ChoiceCriterion{{ID: "a", Description: "a"}, {ID: "b", Description: "b"}}, Target: accept}},
		Fallback: fallback, Policy: policy,
	}, func(ctx agent.Context, _ laya.State, _ []laya.Question) (laya.Prediction, error) {
		<-ctx.Done()
		return laya.Prediction{}, ctx.Err()
	})
	if err != nil {
		t.Fatal(err)
	}
	events, runErr := runRouting(t, routing, routingTerminals{})
	if !errors.Is(runErr, context.DeadlineExceeded) {
		t.Fatalf("run error = %v, want deadline exceeded", runErr)
	}
	for _, event := range events {
		if len(event.Routes) != 0 {
			t.Errorf("timed-out run emitted routes %v", event.Routes)
		}
	}
}

func TestRoutingGraphIsReusableAcrossConcurrentSessions(t *testing.T) {
	prediction := mustChoicePrediction(t, "decision", "accept", 0.8, 0.9, 0.1)
	var predictionCalls atomic.Int32
	var successorCalls atomic.Int32
	accept := workflow.NewFunctionNode("accepted", func(_ agent.Context, output string) (string, error) {
		successorCalls.Add(1)
		return output, nil
	}, workflow.NodeConfig{})
	fallback := terminalNode("fallback", "fallback")
	routing, err := newChoiceRouting(ChoiceRoutingConfig[string]{
		NodeOptions: NodeOptions{Name: "route"}, QuestionID: "decision", Instructions: "choose", ProjectState: textProjector,
		Branches: []ChoiceBranch{{Criteria: []laya.ChoiceCriterion{{ID: "accept", Description: "accept"}, {ID: "reject", Description: "reject"}}, Target: accept}},
		Fallback: fallback, Policy: &AcceptancePolicy{MinSelectedProbability: 0.6, MinConfidence: 0.5, MaxActionProbability: 0.2},
	}, func(agent.Context, laya.State, []laya.Question) (laya.Prediction, error) {
		predictionCalls.Add(1)
		return prediction, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	edges := []workflow.Edge{{From: workflow.Start, To: routing.Node()}}
	edges = append(edges, routing.Edges()...)
	a, err := workflowagent.New(workflowagent.Config{Name: "concurrent_routing", Edges: edges})
	if err != nil {
		t.Fatal(err)
	}
	service := session.InMemoryService()
	r, err := runner.New(runner.Config{AppName: "concurrent_routing", Agent: a, SessionService: service})
	if err != nil {
		t.Fatal(err)
	}
	const count = 12
	var wg sync.WaitGroup
	errs := make(chan error, count)
	for i := 0; i < count; i++ {
		sessionID := fmt.Sprintf("session-%d", i)
		if _, err := service.Create(context.Background(), &session.CreateRequest{AppName: "concurrent_routing", UserID: "test", SessionID: sessionID}); err != nil {
			t.Fatal(err)
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			for event, runErr := range r.Run(context.Background(), "test", sessionID, genai.NewContentFromText("state", genai.RoleUser), agent.RunConfig{}) {
				_ = event
				if runErr != nil {
					errs <- runErr
					return
				}
			}
			errs <- nil
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Errorf("concurrent run error = %v", err)
		}
	}
	if predictionCalls.Load() != count || successorCalls.Load() != count {
		t.Errorf("prediction calls = %d, successor calls = %d; want %d each", predictionCalls.Load(), successorCalls.Load(), count)
	}
}
