package adkcontract_test

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
	"google.golang.org/genai"
)

func TestPinnedADKVersion(t *testing.T) {
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller() failed")
	}
	data, err := os.ReadFile(filepath.Join(filepath.Dir(filename), "..", "..", "go.mod"))
	if err != nil {
		t.Fatalf("read go.mod: %v", err)
	}
	if !strings.Contains(string(data), "google.golang.org/adk/v2 v2.4.0") {
		t.Fatal("go.mod does not pin google.golang.org/adk/v2 v2.4.0")
	}
}

func TestDirectEventCarriesOutputAndOneRoute(t *testing.T) {
	node := workflow.NewFunctionNode("router", func(ctx agent.Context, input string) (*session.Event, error) {
		event := session.NewEvent(ctx, ctx.InvocationID())
		event.Output = input + "-output"
		event.Routes = []string{"selected"}
		return event, nil
	}, workflow.NodeConfig{})
	events, err := runAgent(t, newAgent(t, []workflow.Edge{{From: workflow.Start, To: node}}), "direct", context.Background())
	if err != nil {
		t.Fatalf("runAgent() error = %v", err)
	}
	for _, event := range events {
		if output, ok := event.Output.(string); ok && output == "input-output" {
			if len(event.Routes) != 1 || event.Routes[0] != "selected" {
				t.Fatalf("routes = %v, want [selected]", event.Routes)
			}
			return
		}
	}
	t.Fatal("direct function event was not yielded")
}

func TestDuplicateDestinationNeedsMultiRoute(t *testing.T) {
	router := workflow.NewFunctionNode("router", func(ctx agent.Context, input string) (*session.Event, error) {
		event := session.NewEvent(ctx, ctx.InvocationID())
		event.Output = input
		event.Routes = []string{"a"}
		return event, nil
	}, workflow.NodeConfig{})
	target := workflow.NewFunctionNode("target", func(_ agent.Context, input string) (string, error) { return input, nil }, workflow.NodeConfig{})
	edges := []workflow.Edge{
		{From: workflow.Start, To: router},
		{From: router, To: target, Route: workflow.StringRoute("a")},
		{From: router, To: target, Route: workflow.StringRoute("b")},
	}
	if _, err := workflowagent.New(workflowagent.Config{Name: "duplicate", Edges: edges}); err == nil {
		t.Fatal("workflowagent.New() accepted duplicate From/To edges")
	}
	edges = []workflow.Edge{
		{From: workflow.Start, To: router},
		{From: router, To: target, Route: workflow.MultiRoute[string]{"a", "b"}},
	}
	if _, err := workflowagent.New(workflowagent.Config{Name: "multi", Edges: edges}); err != nil {
		t.Fatalf("workflowagent.New() with MultiRoute error = %v", err)
	}
}

func TestExplicitSchemaRejectsRawGoStructInV240(t *testing.T) {
	type output struct {
		Value string `json:"value"`
	}
	schema, err := jsonschema.For[output](nil)
	if err != nil {
		t.Fatal(err)
	}
	node, err := workflow.NewFunctionNodeWithSchema("typed", func(_ agent.Context, _ string) (output, error) {
		return output{Value: "ok"}, nil
	}, nil, schema, workflow.NodeConfig{})
	if err != nil {
		t.Fatal(err)
	}
	_, err = runAgent(t, newAgent(t, []workflow.Edge{{From: workflow.Start, To: node}}), "schema", context.Background())
	if err == nil || !strings.Contains(err.Error(), "cannot validate against a struct") {
		t.Fatalf("run error = %v, want v2.4.0 raw-struct schema limitation", err)
	}
}

func TestNodeConfigRetryAndTimeout(t *testing.T) {
	t.Run("retry", func(t *testing.T) {
		var attempts atomic.Int32
		retry := workflow.DefaultRetryConfig()
		retry.MaxAttempts = 3
		retry.InitialDelay = 0
		retry.MaxDelay = 0
		node := workflow.NewFunctionNode("retry", func(_ agent.Context, input string) (string, error) {
			if attempts.Add(1) < 3 {
				return "", errors.New("transient")
			}
			return input, nil
		}, workflow.NodeConfig{RetryConfig: retry})
		if _, err := runAgent(t, newAgent(t, []workflow.Edge{{From: workflow.Start, To: node}}), "retry", context.Background()); err != nil {
			t.Fatalf("runAgent() error = %v", err)
		}
		if got := attempts.Load(); got != 3 {
			t.Errorf("attempts = %d, want 3", got)
		}
	})

	t.Run("nil retry means one attempt", func(t *testing.T) {
		var attempts atomic.Int32
		want := errors.New("fail")
		node := workflow.NewFunctionNode("once", func(_ agent.Context, _ string) (string, error) {
			attempts.Add(1)
			return "", want
		}, workflow.NodeConfig{})
		_, err := runAgent(t, newAgent(t, []workflow.Edge{{From: workflow.Start, To: node}}), "once", context.Background())
		if !errors.Is(err, want) || attempts.Load() != 1 {
			t.Fatalf("error = %v, attempts = %d", err, attempts.Load())
		}
	})

	t.Run("timeout", func(t *testing.T) {
		node := workflow.NewFunctionNode("timeout", func(ctx agent.Context, _ string) (string, error) {
			<-ctx.Done()
			return "", ctx.Err()
		}, workflow.NodeConfig{Timeout: 20 * time.Millisecond})
		_, err := runAgent(t, newAgent(t, []workflow.Edge{{From: workflow.Start, To: node}}), "timeout", context.Background())
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("error = %v, want context deadline exceeded", err)
		}
	})
}

func TestParentCancellationStopsSuccessor(t *testing.T) {
	entered := make(chan struct{})
	var successorCalls atomic.Int32
	blocking := workflow.NewFunctionNode("blocking", func(ctx agent.Context, _ string) (string, error) {
		close(entered)
		<-ctx.Done()
		return "", ctx.Err()
	}, workflow.NodeConfig{})
	successor := workflow.NewFunctionNode("successor", func(_ agent.Context, input string) (string, error) {
		successorCalls.Add(1)
		return input, nil
	}, workflow.NodeConfig{})
	a := newAgent(t, workflow.Chain(workflow.Start, blocking, successor))
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := runAgent(t, a, "cancel", ctx)
		done <- err
	}()
	select {
	case <-entered:
		cancel()
	case <-time.After(time.Second):
		t.Fatal("blocking node did not start")
	}
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("error = %v, want context canceled", err)
		}
	case <-time.After(time.Second):
		t.Fatal("cancelled run did not stop")
	}
	if got := successorCalls.Load(); got != 0 {
		t.Errorf("successor calls = %d, want 0", got)
	}
}

func TestWorkflowAgentIsReusableAcrossConcurrentSessions(t *testing.T) {
	var calls atomic.Int32
	node := workflow.NewFunctionNode("shared", func(_ agent.Context, input string) (string, error) {
		calls.Add(1)
		return input, nil
	}, workflow.NodeConfig{})
	a := newAgent(t, []workflow.Edge{{From: workflow.Start, To: node}})
	const count = 12
	var wg sync.WaitGroup
	errs := make(chan error, count)
	for i := 0; i < count; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, err := runAgent(t, a, fmt.Sprintf("concurrent-%d", i), context.Background())
			errs <- err
		}(i)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Errorf("concurrent run error = %v", err)
		}
	}
	if got := calls.Load(); got != count {
		t.Errorf("calls = %d, want %d", got, count)
	}
}

func newAgent(t *testing.T, edges []workflow.Edge) agent.Agent {
	t.Helper()
	a, err := workflowagent.New(workflowagent.Config{Name: "contract_workflow", Edges: edges})
	if err != nil {
		t.Fatalf("workflowagent.New() error = %v", err)
	}
	return a
}

func runAgent(t *testing.T, a agent.Agent, sessionID string, ctx context.Context) ([]*session.Event, error) {
	t.Helper()
	service := session.InMemoryService()
	r, err := runner.New(runner.Config{AppName: "adk_contract", Agent: a, SessionService: service})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}
	if _, err := service.Create(ctx, &session.CreateRequest{AppName: "adk_contract", UserID: "test", SessionID: sessionID}); err != nil {
		return nil, err
	}
	var events []*session.Event
	for event, runErr := range r.Run(ctx, "test", sessionID, genai.NewContentFromText("input", genai.RoleUser), agent.RunConfig{}) {
		if runErr != nil {
			return events, runErr
		}
		events = append(events, event)
	}
	return events, nil
}
