package adklaya

import (
	"fmt"
	"math"
	"strings"

	"github.com/metalagman/laya-go"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

const fallbackRoute = "laya.fallback"

// AcceptancePolicy prevents uncertain predictions from selecting a branch.
// Every threshold is inclusive and must be within [0, 1].
type AcceptancePolicy struct {
	MinSelectedProbability float64
	MinConfidence          float64
	MaxActionProbability   float64
}

// ChoiceBranch maps one or more criteria to one target node.
type ChoiceBranch struct {
	Criteria []laya.ChoiceCriterion
	Target   workflow.Node
}

// ChoiceRoutingConfig configures deterministic choice-based routing.
type ChoiceRoutingConfig[IN any] struct {
	NodeOptions
	QuestionID   laya.QuestionID
	Instructions string
	ProjectState StateProjector[IN]
	Branches     []ChoiceBranch
	Fallback     workflow.Node
	Policy       *AcceptancePolicy
}

// ChoiceRouting contains a routing node and its outgoing conditional edges.
// The routing node forwards its typed domain input unchanged.
type ChoiceRouting struct {
	node  workflow.Node
	edges []workflow.Edge
}

// Node returns the node that evaluates and routes a choice.
func (r *ChoiceRouting) Node() workflow.Node {
	if r == nil {
		return nil
	}
	return r.node
}

// Edges returns a copy of the outgoing conditional edges.
func (r *ChoiceRouting) Edges() []workflow.Edge {
	if r == nil {
		return nil
	}
	edges := append([]workflow.Edge(nil), r.edges...)
	for i := range edges {
		if routes, ok := edges[i].Route.(workflow.MultiRoute[string]); ok {
			edges[i].Route = append(workflow.MultiRoute[string](nil), routes...)
		}
	}
	return edges
}

// NewChoiceRouting constructs one typed choice node and deterministic outgoing
// edges. The supplied model remains caller-owned.
func NewChoiceRouting[IN any](model *laya.Model, cfg ChoiceRoutingConfig[IN]) (*ChoiceRouting, error) {
	if model == nil {
		return nil, invalidConfig("new choice routing: model is nil")
	}
	return newChoiceRouting(cfg, func(ctx agent.Context, state laya.State, questions []laya.Question) (laya.Prediction, error) {
		return model.Predict(ctx, state, questions)
	})
}

func newChoiceRouting[IN any](cfg ChoiceRoutingConfig[IN], predict predictFunc) (*ChoiceRouting, error) {
	if err := validateNodeOptions(cfg.NodeOptions); err != nil {
		return nil, fmt.Errorf("new choice routing: %w", err)
	}
	if cfg.ProjectState == nil {
		return nil, invalidConfig("new choice routing %q: state projector is nil", cfg.Name)
	}
	if cfg.Policy == nil {
		return nil, invalidConfig("new choice routing %q: acceptance policy is nil", cfg.Name)
	}
	policy := *cfg.Policy
	if err := validatePolicy(policy); err != nil {
		return nil, invalidConfig("new choice routing %q: %v", cfg.Name, err)
	}
	if cfg.Fallback == nil {
		return nil, invalidConfig("new choice routing %q: fallback target is nil", cfg.Name)
	}
	if len(cfg.Branches) == 0 {
		return nil, invalidConfig("new choice routing %q: branches are empty", cfg.Name)
	}

	criteria := make([]laya.ChoiceCriterion, 0)
	criterionRoute := make(map[laya.CriterionID]string)
	targetRoutes := make(map[workflow.Node][]string)
	targetNames := make(map[string]workflow.Node)
	targetOrder := make([]workflow.Node, 0, len(cfg.Branches)+1)
	for i, branch := range cfg.Branches {
		if branch.Target == nil {
			return nil, invalidConfig("new choice routing %q: branch %d target is nil", cfg.Name, i)
		}
		if len(branch.Criteria) == 0 {
			return nil, invalidConfig("new choice routing %q: branch %d criteria are empty", cfg.Name, i)
		}
		if err := validateTarget(cfg.Name, branch.Target, targetNames); err != nil {
			return nil, err
		}
		route := fmt.Sprintf("laya.branch.%d", i)
		if _, ok := targetRoutes[branch.Target]; !ok {
			targetOrder = append(targetOrder, branch.Target)
		}
		targetRoutes[branch.Target] = append(targetRoutes[branch.Target], route)
		for _, criterion := range branch.Criteria {
			if _, exists := criterionRoute[criterion.ID]; exists {
				return nil, invalidConfig("new choice routing %q: criterion %q appears in multiple branches", cfg.Name, criterion.ID)
			}
			criterionRoute[criterion.ID] = route
			criteria = append(criteria, criterion)
		}
	}
	if _, exists := targetRoutes[cfg.Fallback]; exists {
		return nil, invalidConfig("new choice routing %q: fallback target is also a branch target", cfg.Name)
	}
	if err := validateTarget(cfg.Name, cfg.Fallback, targetNames); err != nil {
		return nil, err
	}
	question, err := laya.NewChoiceQuestion(cfg.QuestionID, cfg.Instructions, criteria)
	if err != nil {
		return nil, fmt.Errorf("new choice routing %q: %w", cfg.Name, err)
	}
	fn := func(ctx agent.Context, input IN) (*session.Event, error) {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		state, err := cfg.ProjectState(ctx, input)
		if err != nil {
			return nil, fmt.Errorf("project state: %w", err)
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		prediction, err := predict(ctx, state, []laya.Question{question})
		if err != nil {
			return nil, err
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		result, ok := prediction.Results()[0].(laya.ChoiceResult)
		if !ok {
			return nil, fmt.Errorf("choice routing %q: result has type %T", cfg.Name, prediction.Results()[0])
		}
		route := acceptedRoute(result, criterionRoute, policy)
		event := session.NewEvent(ctx, ctx.InvocationID())
		event.Output = input
		event.Routes = []string{route}
		return event, nil
	}
	node := workflow.NewFunctionNode(cfg.Name, fn, cloneNodeConfig(cfg.Config))
	edges := make([]workflow.Edge, 0, len(targetOrder)+1)
	for _, target := range targetOrder {
		edges = append(edges, workflow.Edge{From: node, To: target, Route: workflow.MultiRoute[string](append([]string(nil), targetRoutes[target]...))})
	}
	edges = append(edges, workflow.Edge{From: node, To: cfg.Fallback, Route: workflow.StringRoute(fallbackRoute)})
	return &ChoiceRouting{node: node, edges: edges}, nil
}

func validateTarget(routingName string, target workflow.Node, names map[string]workflow.Node) error {
	name := target.Name()
	if strings.TrimSpace(name) == "" {
		return invalidConfig("new choice routing %q: target name is empty", routingName)
	}
	if name == "user" {
		return invalidConfig("new choice routing %q: target name %q is reserved by ADK", routingName, name)
	}
	if existing, ok := names[name]; ok && existing != target {
		return invalidConfig("new choice routing %q: distinct targets repeat name %q", routingName, name)
	}
	names[name] = target
	return nil
}

func acceptedRoute(result laya.ChoiceResult, routes map[laya.CriterionID]string, policy AcceptancePolicy) string {
	probabilities := result.Probabilities()
	max := -1.0
	maxCount := 0
	selectedProbability := -1.0
	for _, probability := range probabilities {
		if probability.Probability > max {
			max = probability.Probability
			maxCount = 1
		} else if probability.Probability == max {
			maxCount++
		}
		if probability.CriterionID == result.Selected() {
			selectedProbability = probability.Probability
		}
	}
	if maxCount != 1 || selectedProbability != max || selectedProbability < policy.MinSelectedProbability || result.Confidence() < policy.MinConfidence || result.ActionProbability() > policy.MaxActionProbability {
		return fallbackRoute
	}
	if route, ok := routes[result.Selected()]; ok {
		return route
	}
	return fallbackRoute
}

func validatePolicy(policy AcceptancePolicy) error {
	values := []struct {
		name  string
		value float64
	}{
		{"minimum selected probability", policy.MinSelectedProbability},
		{"minimum confidence", policy.MinConfidence},
		{"maximum action probability", policy.MaxActionProbability},
	}
	for _, value := range values {
		if math.IsNaN(value.value) || math.IsInf(value.value, 0) || value.value < 0 || value.value > 1 {
			return fmt.Errorf("%s %v is outside [0,1]", value.name, value.value)
		}
	}
	return nil
}
