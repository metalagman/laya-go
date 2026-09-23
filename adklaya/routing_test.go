package adklaya

import (
	"context"
	"errors"
	"math"
	"sync"
	"testing"

	"github.com/metalagman/laya-go"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
	"google.golang.org/genai"
)

func TestChoiceRoutingAcceptance(t *testing.T) {
	tests := []struct {
		name                string
		selected            laya.CriterionID
		selectedProbability float64
		confidence          float64
		action              float64
		tie                 bool
		wantRoute           string
		wantTerminal        string
	}{
		{name: "accepted at inclusive boundaries", selected: "accept", selectedProbability: 0.6, confidence: 0.5, action: 0.2, wantRoute: "laya.branch.0", wantTerminal: "accepted"},
		{name: "other branch", selected: "reject", selectedProbability: 0.8, confidence: 0.9, action: 0.1, wantRoute: "laya.branch.1", wantTerminal: "rejected"},
		{name: "probability below threshold", selected: "accept", selectedProbability: 0.59, confidence: 0.9, action: 0.1, wantRoute: fallbackRoute, wantTerminal: "fallback"},
		{name: "confidence below threshold", selected: "accept", selectedProbability: 0.8, confidence: 0.49, action: 0.1, wantRoute: fallbackRoute, wantTerminal: "fallback"},
		{name: "action above threshold", selected: "accept", selectedProbability: 0.8, confidence: 0.9, action: 0.21, wantRoute: fallbackRoute, wantTerminal: "fallback"},
		{name: "exact maximum tie", selected: "accept", selectedProbability: 0.5, confidence: 0.9, action: 0.1, tie: true, wantRoute: fallbackRoute, wantTerminal: "fallback"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			prediction := mustChoicePrediction(t, "decision", tc.selected, tc.selectedProbability, tc.confidence, tc.action)
			if tc.tie {
				result, err := laya.NewChoiceResult("decision", laya.ChoiceAnswer{Selected: "accept", Probabilities: []laya.CriterionProbability{{CriterionID: "accept", Probability: 0.5}, {CriterionID: "reject", Probability: 0.5}}, Confidence: tc.confidence, ActionProbability: tc.action})
				if err != nil {
					t.Fatal(err)
				}
				prediction, err = laya.NewPrediction([]laya.Result{result}, laya.Usage{}, laya.PredictionMetadata{})
				if err != nil {
					t.Fatal(err)
				}
			}
			routing, terminals := newTestRouting(t, func(agent.Context, laya.State, []laya.Question) (laya.Prediction, error) { return prediction, nil })
			events, err := runRouting(t, routing, terminals)
			if err != nil {
				t.Fatalf("runRouting() error = %v", err)
			}
			var routes []string
			var terminal string
			for _, event := range events {
				if len(event.Routes) > 0 {
					routes = event.Routes
					if output, ok := event.Output.(string); !ok || output != "state" {
						t.Errorf("routing event output = %T(%v), want unchanged domain input", event.Output, event.Output)
					}
				}
				if output, ok := event.Output.(string); ok && (output == "accepted" || output == "rejected" || output == "fallback") {
					terminal = output
				}
			}
			if len(routes) != 1 || routes[0] != tc.wantRoute {
				t.Errorf("routes = %v, want [%q]", routes, tc.wantRoute)
			}
			if terminal != tc.wantTerminal {
				t.Errorf("terminal = %q, want %q", terminal, tc.wantTerminal)
			}
		})
	}
}

func TestChoiceRoutingConsolidatesSharedTargetsAndCopiesEdges(t *testing.T) {
	shared := terminalNode("shared", "shared")
	other := terminalNode("other", "other")
	fallback := terminalNode("fallback", "fallback")
	policy := &AcceptancePolicy{MinSelectedProbability: 0, MinConfidence: 0, MaxActionProbability: 1}
	cfg := ChoiceRoutingConfig[string]{
		NodeOptions: NodeOptions{Name: "route"}, QuestionID: "decision", Instructions: "choose", ProjectState: textProjector,
		Branches: []ChoiceBranch{
			{Criteria: []laya.ChoiceCriterion{{ID: "a", Description: "a"}}, Target: shared},
			{Criteria: []laya.ChoiceCriterion{{ID: "b", Description: "b"}}, Target: shared},
			{Criteria: []laya.ChoiceCriterion{{ID: "c", Description: "c"}}, Target: other},
		},
		Fallback: fallback, Policy: policy,
	}
	routing, err := newChoiceRouting(cfg, func(agent.Context, laya.State, []laya.Question) (laya.Prediction, error) {
		return laya.Prediction{}, errors.New("unused")
	})
	if err != nil {
		t.Fatalf("newChoiceRouting() error = %v", err)
	}
	edges := routing.Edges()
	if len(edges) != 3 {
		t.Fatalf("len(edges) = %d, want shared + other + fallback", len(edges))
	}
	multi, ok := edges[0].Route.(workflow.MultiRoute[string])
	if !ok || len(multi) != 2 {
		t.Fatalf("shared route = %#v, want two-value MultiRoute", edges[0].Route)
	}
	multi[0] = "mutated"
	otherEdges := routing.Edges()
	otherMulti, ok := otherEdges[0].Route.(workflow.MultiRoute[string])
	if !ok || len(otherMulti) != 2 || otherMulti[0] != "laya.branch.0" {
		t.Fatalf("route after caller mutation = %#v, want original routes", otherEdges[0].Route)
	}
	edges[0] = workflow.Edge{}
	if routing.Edges()[0].From == nil {
		t.Error("Edges returned internal backing storage")
	}
	var workers sync.WaitGroup
	for range 8 {
		workers.Go(func() {
			for range 100 {
				copy := routing.Edges()
				routes := copy[0].Route.(workflow.MultiRoute[string])
				routes[0] = "caller-owned"
				if got := routing.Edges()[0].Route.(workflow.MultiRoute[string])[0]; got != "laya.branch.0" {
					t.Errorf("concurrent Edges route = %q, want original", got)
				}
			}
		})
	}
	workers.Wait()
}

func TestChoiceRoutingErrorsDoNotFallback(t *testing.T) {
	want := errors.New("inference failed")
	routing, terminals := newTestRouting(t, func(agent.Context, laya.State, []laya.Question) (laya.Prediction, error) {
		return laya.Prediction{}, want
	})
	events, err := runRouting(t, routing, terminals)
	if !errors.Is(err, want) {
		t.Fatalf("runRouting() error = %v, want errors.Is(_, %v)", err, want)
	}
	for _, event := range events {
		if len(event.Routes) != 0 {
			t.Errorf("error emitted routes %v", event.Routes)
		}
		if output, ok := event.Output.(string); ok && output == "fallback" {
			t.Error("error executed fallback")
		}
	}
}

func TestChoiceRoutingRejectsInvalidConfiguration(t *testing.T) {
	target := terminalNode("target", "target")
	duplicateName := terminalNode("target", "other")
	base := ChoiceRoutingConfig[string]{
		NodeOptions: NodeOptions{Name: "route"}, QuestionID: "decision", Instructions: "choose", ProjectState: textProjector,
		Branches: []ChoiceBranch{{Criteria: []laya.ChoiceCriterion{{ID: "a", Description: "a"}, {ID: "b", Description: "b"}}, Target: target}},
		Fallback: terminalNode("fallback", "fallback"), Policy: &AcceptancePolicy{MaxActionProbability: 1},
	}
	predict := func(agent.Context, laya.State, []laya.Question) (laya.Prediction, error) {
		return laya.Prediction{}, nil
	}
	tests := []struct {
		name   string
		mutate func(*ChoiceRoutingConfig[string])
	}{
		{name: "nil policy", mutate: func(c *ChoiceRoutingConfig[string]) { c.Policy = nil }},
		{name: "non-finite policy", mutate: func(c *ChoiceRoutingConfig[string]) { c.Policy = &AcceptancePolicy{MinConfidence: math.NaN()} }},
		{name: "no branches", mutate: func(c *ChoiceRoutingConfig[string]) { c.Branches = nil }},
		{name: "nil fallback", mutate: func(c *ChoiceRoutingConfig[string]) { c.Fallback = nil }},
		{name: "fallback reused", mutate: func(c *ChoiceRoutingConfig[string]) { c.Fallback = target }},
		{name: "duplicate target name", mutate: func(c *ChoiceRoutingConfig[string]) {
			c.Branches = []ChoiceBranch{
				{Criteria: []laya.ChoiceCriterion{{ID: "a", Description: "a"}}, Target: target},
				{Criteria: []laya.ChoiceCriterion{{ID: "b", Description: "b"}}, Target: duplicateName},
			}
		}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := base
			tc.mutate(&cfg)
			if _, err := newChoiceRouting(cfg, predict); err == nil {
				t.Error("newChoiceRouting() succeeded, want error")
			}
		})
	}
}

type routingTerminals struct {
	accept   workflow.Node
	reject   workflow.Node
	fallback workflow.Node
}

func newTestRouting(t *testing.T, predict predictFunc) (*ChoiceRouting, routingTerminals) {
	t.Helper()
	terminals := routingTerminals{accept: terminalNode("accepted", "accepted"), reject: terminalNode("rejected", "rejected"), fallback: terminalNode("fallback", "fallback")}
	routing, err := newChoiceRouting(ChoiceRoutingConfig[string]{
		NodeOptions: NodeOptions{Name: "route"}, QuestionID: "decision", Instructions: "choose", ProjectState: textProjector,
		Branches: []ChoiceBranch{
			{Criteria: []laya.ChoiceCriterion{{ID: "accept", Description: "accept"}}, Target: terminals.accept},
			{Criteria: []laya.ChoiceCriterion{{ID: "reject", Description: "reject"}}, Target: terminals.reject},
		},
		Fallback: terminals.fallback,
		Policy:   &AcceptancePolicy{MinSelectedProbability: 0.6, MinConfidence: 0.5, MaxActionProbability: 0.2},
	}, predict)
	if err != nil {
		t.Fatalf("newChoiceRouting() error = %v", err)
	}
	return routing, terminals
}

func runRouting(t *testing.T, routing *ChoiceRouting, _ routingTerminals) ([]*session.Event, error) {
	t.Helper()
	edges := []workflow.Edge{{From: workflow.Start, To: routing.Node()}}
	edges = append(edges, routing.Edges()...)
	a, err := workflowagent.New(workflowagent.Config{Name: "routing_workflow", Edges: edges})
	if err != nil {
		t.Fatalf("workflowagent.New() error = %v", err)
	}
	service := session.InMemoryService()
	r, err := runner.New(runner.Config{AppName: "routing_test", Agent: a, SessionService: service})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}
	ctx := context.Background()
	if _, err := service.Create(ctx, &session.CreateRequest{AppName: "routing_test", UserID: "test", SessionID: "session"}); err != nil {
		t.Fatal(err)
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

func terminalNode(name, output string) workflow.Node {
	return workflow.NewFunctionNode(name, func(_ agent.Context, _ string) (string, error) { return output, nil }, workflow.NodeConfig{})
}

func textProjector(_ agent.Context, input string) (laya.State, error) {
	return laya.TextState(input), nil
}
