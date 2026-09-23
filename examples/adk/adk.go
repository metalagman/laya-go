// Package adk demonstrates application-owned Google ADK runner composition
// over one already-open, shared Laya model.
package adk

import (
	"context"
	"encoding/json"
	"fmt"
	"io"

	"github.com/metalagman/laya-go"
	"github.com/metalagman/laya-go/adklaya"
	"github.com/metalagman/laya-go/examples/internal/exampleapp"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
	"google.golang.org/genai"
)

// Report is a content-safe summary from data and routing graphs.
type Report struct {
	Results  int           `json:"results"`
	Route    string        `json:"route"`
	BundleID laya.BundleID `json:"bundleId"`
}

// Run executes typed data and deterministic routing graphs over model. The
// caller owns model and must keep it open until Run returns.
func Run(ctx context.Context, model *laya.Model, input string, policy adklaya.AcceptancePolicy, output io.Writer) error {
	if model == nil || output == nil {
		return fmt.Errorf("run ADK example: %w: model and output are required", laya.ErrInvalidConfig)
	}
	questions, err := exampleapp.Questions()
	if err != nil {
		return err
	}
	project := func(_ agent.Context, value string) (laya.State, error) { return laya.TextState(value), nil }
	data, err := adklaya.NewPredictionNode(model, project, questions, adklaya.NodeOptions{Name: "laya_data"})
	if err != nil {
		return fmt.Errorf("create ADK data node: %w", err)
	}
	dataEvents, err := run(ctx, "example_data", []workflow.Edge{{From: workflow.Start, To: data}}, input)
	if err != nil {
		return fmt.Errorf("run ADK data graph: %w", err)
	}
	var prediction adklaya.PredictionOutput
	for _, event := range dataEvents {
		if value, ok := event.Output.(adklaya.PredictionOutput); ok {
			prediction = value
		}
	}
	if len(prediction.Results) != len(questions) {
		return fmt.Errorf("ADK data graph: %w: missing prediction output", laya.ErrInvalidOutput)
	}

	accepted := terminal("accepted")
	fallback := terminal("fallback")
	choice := questions[0].(laya.ChoiceQuestion)
	criteria := choice.Criteria()
	routing, err := adklaya.NewChoiceRouting(model, adklaya.ChoiceRoutingConfig[string]{
		NodeOptions: adklaya.NodeOptions{Name: "laya_route"}, QuestionID: choice.ID(), Instructions: choice.Instructions(), ProjectState: project,
		Branches: []adklaya.ChoiceBranch{{Criteria: criteria, Target: accepted}},
		Fallback: fallback, Policy: &policy,
	})
	if err != nil {
		return fmt.Errorf("create ADK routing node: %w", err)
	}
	edges := []workflow.Edge{{From: workflow.Start, To: routing.Node()}}
	edges = append(edges, routing.Edges()...)
	routeEvents, err := run(ctx, "example_route", edges, input)
	if err != nil {
		return fmt.Errorf("run ADK routing graph: %w", err)
	}
	var selected string
	for _, event := range routeEvents {
		if value, ok := event.Output.(string); ok && (value == "accepted" || value == "fallback") {
			selected = value
		}
	}
	if selected == "" {
		return fmt.Errorf("ADK routing graph: %w: successor did not run", laya.ErrInvalidOutput)
	}
	if err := json.NewEncoder(output).Encode(Report{Results: len(prediction.Results), Route: selected, BundleID: prediction.Metadata.BundleID}); err != nil {
		return fmt.Errorf("encode ADK example report: %w", err)
	}
	return nil
}

func terminal(output string) workflow.Node {
	return workflow.NewFunctionNode(output, func(_ agent.Context, _ string) (string, error) { return output, nil }, workflow.NodeConfig{})
}

func run(ctx context.Context, name string, edges []workflow.Edge, input string) ([]*session.Event, error) {
	a, err := workflowagent.New(workflowagent.Config{Name: name, Edges: edges})
	if err != nil {
		return nil, err
	}
	service := session.InMemoryService()
	r, err := runner.New(runner.Config{AppName: name, Agent: a, SessionService: service})
	if err != nil {
		return nil, err
	}
	if _, err := service.Create(ctx, &session.CreateRequest{AppName: name, UserID: "example", SessionID: "one"}); err != nil {
		return nil, err
	}
	var events []*session.Event
	for event, runErr := range r.Run(ctx, "example", "one", genai.NewContentFromText(input, genai.RoleUser), agent.RunConfig{}) {
		if runErr != nil {
			return events, runErr
		}
		events = append(events, event)
	}
	return events, nil
}
