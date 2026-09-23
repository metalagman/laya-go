//go:build laya_native && cgo

package adklaya_test

import (
	"context"
	"os"
	"testing"

	"github.com/metalagman/laya-go"
	"github.com/metalagman/laya-go/adklaya"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
	"google.golang.org/genai"
)

const protectedBundleID = laya.BundleID("sha256:963bc035d885e0463ea1d7d54906cadf0e2a3a41073e131921800ea5cc358935")

func TestProtectedNativeADKGraphs(t *testing.T) {
	bundleDir := os.Getenv("LAYA_BUNDLE_DIR")
	if bundleDir == "" || os.Getenv("LAYA_ONNXRUNTIME_LIBRARY") == "" {
		t.Skip("protected native artifacts were not supplied")
	}
	runtime, err := laya.NewRuntime(t.Context(), laya.RuntimeOptions{})
	if err != nil {
		t.Fatalf("NewRuntime() error = %v", err)
	}
	model, err := runtime.OpenModelDir(t.Context(), bundleDir, laya.ModelOptions{Truncation: laya.TruncateOverflow, QueueCapacity: 2})
	if err != nil {
		_ = runtime.Close(t.Context())
		t.Fatalf("OpenModelDir() error = %v", err)
	}
	t.Cleanup(func() {
		if err := model.Close(context.Background()); err != nil {
			t.Errorf("Model.Close() error = %v", err)
		}
		if err := runtime.Close(context.Background()); err != nil {
			t.Errorf("Runtime.Close() error = %v", err)
		}
	})

	choice, err := laya.NewChoiceQuestion("route", "Choose the safest route", []laya.ChoiceCriterion{
		{ID: "safe", Description: "lowest risk"},
		{ID: "fast", Description: "lowest latency"},
		{ID: "cheap", Description: "lowest cost"},
	})
	if err != nil {
		t.Fatal(err)
	}
	score, err := laya.NewScoreQuestion("urgency", "Оцени срочность", []laya.ScoreLevel{
		{Label: "не срочно", Description: "не срочно"},
		{Label: "обычно", Description: "обычно"},
		{Label: "срочно", Description: "срочно"},
		{Label: "критично", Description: "критично"},
	})
	if err != nil {
		t.Fatal(err)
	}
	noul, err := laya.NewNoulQuestion("acceptable", "Is <mask> маршрут acceptable?", "no", "yes")
	if err != nil {
		t.Fatal(err)
	}
	project := func(_ agent.Context, input string) (laya.State, error) { return laya.TextState(input), nil }
	mixed, err := adklaya.NewPredictionNode(model, project, []laya.Question{choice, score, noul}, adklaya.NodeOptions{Name: "mixed"})
	if err != nil {
		t.Fatalf("NewPredictionNode() error = %v", err)
	}
	events := runProtectedGraph(t, "mixed_graph", []workflow.Edge{{From: workflow.Start, To: mixed}}, `{"emoji":"🧭","text":"mixed Русский + English","n":1} <mask>`)
	var mixedOutput adklaya.PredictionOutput
	for _, event := range events {
		if output, ok := event.Output.(adklaya.PredictionOutput); ok {
			mixedOutput = output
		}
	}
	if len(mixedOutput.Results) != 3 || mixedOutput.Results[0].Kind != adklaya.ResultKindChoice || mixedOutput.Results[1].Kind != adklaya.ResultKindScore || mixedOutput.Results[2].Kind != adklaya.ResultKindNoul {
		t.Fatalf("mixed results = %#v", mixedOutput.Results)
	}
	if mixedOutput.Metadata.BundleID != protectedBundleID || mixedOutput.Metadata.RuntimeID == "" {
		t.Errorf("mixed metadata = %#v", mixedOutput.Metadata)
	}

	accepted := terminal("accepted", "accepted")
	fallback := terminal("fallback", "fallback")
	routing, err := adklaya.NewChoiceRouting(model, adklaya.ChoiceRoutingConfig[string]{
		NodeOptions: adklaya.NodeOptions{Name: "route"}, QuestionID: "route", Instructions: "Choose the safest route", ProjectState: project,
		Branches: []adklaya.ChoiceBranch{
			{Criteria: []laya.ChoiceCriterion{{ID: "safe", Description: "lowest risk"}}, Target: accepted},
			{Criteria: []laya.ChoiceCriterion{{ID: "fast", Description: "lowest latency"}}, Target: accepted},
			{Criteria: []laya.ChoiceCriterion{{ID: "cheap", Description: "lowest cost"}}, Target: accepted},
		},
		Fallback: fallback, Policy: &adklaya.AcceptancePolicy{MaxActionProbability: 1},
	})
	if err != nil {
		t.Fatalf("NewChoiceRouting() error = %v", err)
	}
	edges := []workflow.Edge{{From: workflow.Start, To: routing.Node()}}
	edges = append(edges, routing.Edges()...)
	events = runProtectedGraph(t, "accepted_graph", edges, `{"emoji":"🧭","text":"mixed Русский + English","n":1}`)
	assertProtectedRoute(t, events, "accepted")

	strictRouting, err := adklaya.NewChoiceRouting(model, adklaya.ChoiceRoutingConfig[string]{
		NodeOptions: adklaya.NodeOptions{Name: "strict_route"}, QuestionID: "route", Instructions: "Choose the safest route", ProjectState: project,
		Branches: []adklaya.ChoiceBranch{{Criteria: []laya.ChoiceCriterion{{ID: "safe", Description: "lowest risk"}, {ID: "fast", Description: "lowest latency"}, {ID: "cheap", Description: "lowest cost"}}, Target: accepted}},
		Fallback: fallback, Policy: &adklaya.AcceptancePolicy{MinSelectedProbability: 1, MinConfidence: 1, MaxActionProbability: 0},
	})
	if err != nil {
		t.Fatalf("NewChoiceRouting(strict) error = %v", err)
	}
	edges = []workflow.Edge{{From: workflow.Start, To: strictRouting.Node()}}
	edges = append(edges, strictRouting.Edges()...)
	events = runProtectedGraph(t, "fallback_graph", edges, `{"emoji":"🧭","text":"mixed Русский + English","n":1}`)
	assertProtectedRoute(t, events, "fallback")
}

func runProtectedGraph(t *testing.T, name string, edges []workflow.Edge, input string) []*session.Event {
	t.Helper()
	a, err := workflowagent.New(workflowagent.Config{Name: name, Edges: edges})
	if err != nil {
		t.Fatalf("workflowagent.New() error = %v", err)
	}
	service := session.InMemoryService()
	r, err := runner.New(runner.Config{AppName: name, Agent: a, SessionService: service})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}
	if _, err := service.Create(t.Context(), &session.CreateRequest{AppName: name, UserID: "test", SessionID: "session"}); err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	var events []*session.Event
	for event, runErr := range r.Run(t.Context(), "test", "session", genai.NewContentFromText(input, genai.RoleUser), agent.RunConfig{}) {
		if runErr != nil {
			t.Fatalf("Run() error = %v", runErr)
		}
		events = append(events, event)
	}
	return events
}

func terminal(name, output string) workflow.Node {
	return workflow.NewFunctionNode(name, func(_ agent.Context, _ string) (string, error) { return output, nil }, workflow.NodeConfig{})
}

func assertProtectedRoute(t *testing.T, events []*session.Event, terminalOutput string) {
	t.Helper()
	var routeCount, terminalCount int
	for _, event := range events {
		if len(event.Routes) > 0 {
			routeCount++
			if len(event.Routes) != 1 {
				t.Errorf("routes = %v, want exactly one", event.Routes)
			}
			if output, ok := event.Output.(string); !ok || output == "" {
				t.Errorf("routing output = %#v, want unchanged domain string", event.Output)
			}
		}
		if output, ok := event.Output.(string); ok && output == terminalOutput {
			terminalCount++
		}
	}
	if routeCount != 1 || terminalCount != 1 {
		t.Errorf("route events = %d, %q terminal events = %d; want 1 each", routeCount, terminalOutput, terminalCount)
	}
}
