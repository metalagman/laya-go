package adklaya

import (
	"fmt"
	"strings"

	"github.com/metalagman/laya-go"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/workflow"
)

// StateProjector converts a workflow value into the state evaluated by Laya.
type StateProjector[IN any] func(agent.Context, IN) (laya.State, error)

// NodeOptions configures an ADK node. Name must be non-empty and cannot be
// "user", which ADK reserves for end-user input.
type NodeOptions struct {
	Name   string
	Config workflow.NodeConfig
}

// ChoiceOutput is the schema-friendly result of one choice question.
type ChoiceOutput struct {
	QuestionID laya.QuestionID         `json:"questionId"`
	Answer     laya.ChoiceAnswer       `json:"answer"`
	Usage      laya.Usage              `json:"usage"`
	Metadata   laya.PredictionMetadata `json:"metadata"`
}

// ScoreOutput is the schema-friendly result of one score question.
type ScoreOutput struct {
	QuestionID laya.QuestionID         `json:"questionId"`
	Answer     laya.ScoreAnswer        `json:"answer"`
	Usage      laya.Usage              `json:"usage"`
	Metadata   laya.PredictionMetadata `json:"metadata"`
}

// NoulOutput is the schema-friendly result of one boolean question.
type NoulOutput struct {
	QuestionID laya.QuestionID         `json:"questionId"`
	Answer     laya.NoulAnswer         `json:"answer"`
	Usage      laya.Usage              `json:"usage"`
	Metadata   laya.PredictionMetadata `json:"metadata"`
}

// ResultKind discriminates a ResultOutput value.
type ResultKind string

const (
	// ResultKindChoice identifies Choice.
	ResultKindChoice ResultKind = "choice"
	// ResultKindScore identifies Score.
	ResultKindScore ResultKind = "score"
	// ResultKindNoul identifies Noul.
	ResultKindNoul ResultKind = "noul"
)

// ResultOutput is one ordered result in a PredictionOutput. Exactly one
// answer pointer is non-nil and agrees with Kind.
type ResultOutput struct {
	Kind       ResultKind         `json:"kind"`
	QuestionID laya.QuestionID    `json:"questionId"`
	Choice     *laya.ChoiceAnswer `json:"choice,omitempty"`
	Score      *laya.ScoreAnswer  `json:"score,omitempty"`
	Noul       *laya.NoulAnswer   `json:"noul,omitempty"`
}

// PredictionOutput contains ordered mixed results and batch-level evidence.
type PredictionOutput struct {
	Results  []ResultOutput          `json:"results"`
	Usage    laya.Usage              `json:"usage"`
	Metadata laya.PredictionMetadata `json:"metadata"`
}

type predictFunc func(agent.Context, laya.State, []laya.Question) (laya.Prediction, error)

// NewChoiceNode constructs a typed data node for one choice question.
func NewChoiceNode[IN any](model *laya.Model, projector StateProjector[IN], question laya.ChoiceQuestion, opts NodeOptions) (*workflow.FunctionNode, error) {
	if model == nil {
		return nil, invalidConfig("new choice node: model is nil")
	}
	return newChoiceNode(projector, question, opts, func(ctx agent.Context, state laya.State, questions []laya.Question) (laya.Prediction, error) {
		return model.Predict(ctx, state, questions)
	})
}

func newChoiceNode[IN any](projector StateProjector[IN], question laya.ChoiceQuestion, opts NodeOptions, predict predictFunc) (*workflow.FunctionNode, error) {
	return newDataNode(opts, projector, func(ctx agent.Context, state laya.State) (ChoiceOutput, error) {
		prediction, err := predict(ctx, state, []laya.Question{question})
		if err != nil {
			return ChoiceOutput{}, err
		}
		result, ok := prediction.Results()[0].(laya.ChoiceResult)
		if !ok {
			return ChoiceOutput{}, fmt.Errorf("choice node %q: result has type %T", opts.Name, prediction.Results()[0])
		}
		return ChoiceOutput{
			QuestionID: result.QuestionID(),
			Answer: laya.ChoiceAnswer{
				Selected:          result.Selected(),
				Probabilities:     result.Probabilities(),
				Confidence:        result.Confidence(),
				ActionProbability: result.ActionProbability(),
			},
			Usage: prediction.Usage(), Metadata: prediction.Metadata(),
		}, nil
	})
}

// NewScoreNode constructs a typed data node for one ordinal score question.
func NewScoreNode[IN any](model *laya.Model, projector StateProjector[IN], question laya.ScoreQuestion, opts NodeOptions) (*workflow.FunctionNode, error) {
	if model == nil {
		return nil, invalidConfig("new score node: model is nil")
	}
	return newScoreNode(projector, question, opts, func(ctx agent.Context, state laya.State, questions []laya.Question) (laya.Prediction, error) {
		return model.Predict(ctx, state, questions)
	})
}

func newScoreNode[IN any](projector StateProjector[IN], question laya.ScoreQuestion, opts NodeOptions, predict predictFunc) (*workflow.FunctionNode, error) {
	return newDataNode(opts, projector, func(ctx agent.Context, state laya.State) (ScoreOutput, error) {
		prediction, err := predict(ctx, state, []laya.Question{question})
		if err != nil {
			return ScoreOutput{}, err
		}
		result, ok := prediction.Results()[0].(laya.ScoreResult)
		if !ok {
			return ScoreOutput{}, fmt.Errorf("score node %q: result has type %T", opts.Name, prediction.Results()[0])
		}
		return ScoreOutput{
			QuestionID: result.QuestionID(),
			Answer: laya.ScoreAnswer{
				ExpectedLevel:     result.ExpectedLevel(),
				Distribution:      result.Distribution(),
				Confidence:        result.Confidence(),
				ActionProbability: result.ActionProbability(),
			},
			Usage: prediction.Usage(), Metadata: prediction.Metadata(),
		}, nil
	})
}

// NewNoulNode constructs a typed data node for one boolean question.
func NewNoulNode[IN any](model *laya.Model, projector StateProjector[IN], question laya.NoulQuestion, opts NodeOptions) (*workflow.FunctionNode, error) {
	if model == nil {
		return nil, invalidConfig("new noul node: model is nil")
	}
	return newNoulNode(projector, question, opts, func(ctx agent.Context, state laya.State, questions []laya.Question) (laya.Prediction, error) {
		return model.Predict(ctx, state, questions)
	})
}

func newNoulNode[IN any](projector StateProjector[IN], question laya.NoulQuestion, opts NodeOptions, predict predictFunc) (*workflow.FunctionNode, error) {
	return newDataNode(opts, projector, func(ctx agent.Context, state laya.State) (NoulOutput, error) {
		prediction, err := predict(ctx, state, []laya.Question{question})
		if err != nil {
			return NoulOutput{}, err
		}
		result, ok := prediction.Results()[0].(laya.NoulResult)
		if !ok {
			return NoulOutput{}, fmt.Errorf("noul node %q: result has type %T", opts.Name, prediction.Results()[0])
		}
		return NoulOutput{
			QuestionID: result.QuestionID(),
			Answer: laya.NoulAnswer{
				TrueProbability:   result.TrueProbability(),
				ActionProbability: result.ActionProbability(),
			},
			Usage: prediction.Usage(), Metadata: prediction.Metadata(),
		}, nil
	})
}

// NewPredictionNode constructs a typed data node for an ordered mixed batch.
func NewPredictionNode[IN any](model *laya.Model, projector StateProjector[IN], questions []laya.Question, opts NodeOptions) (*workflow.FunctionNode, error) {
	if model == nil {
		return nil, invalidConfig("new prediction node: model is nil")
	}
	return newPredictionNode(projector, questions, opts, func(ctx agent.Context, state laya.State, questions []laya.Question) (laya.Prediction, error) {
		return model.Predict(ctx, state, questions)
	})
}

func newPredictionNode[IN any](projector StateProjector[IN], questions []laya.Question, opts NodeOptions, predict predictFunc) (*workflow.FunctionNode, error) {
	questions = append([]laya.Question(nil), questions...)
	if len(questions) == 0 {
		return nil, invalidConfig("new prediction node: questions are empty")
	}
	return newDataNode(opts, projector, func(ctx agent.Context, state laya.State) (PredictionOutput, error) {
		prediction, err := predict(ctx, state, questions)
		if err != nil {
			return PredictionOutput{}, err
		}
		results := prediction.Results()
		output := PredictionOutput{Results: make([]ResultOutput, 0, len(results)), Usage: prediction.Usage(), Metadata: prediction.Metadata()}
		for _, result := range results {
			converted, err := resultOutput(result)
			if err != nil {
				return PredictionOutput{}, fmt.Errorf("prediction node %q: %w", opts.Name, err)
			}
			output.Results = append(output.Results, converted)
		}
		return output, nil
	})
}

func newDataNode[IN, OUT any](opts NodeOptions, projector StateProjector[IN], run func(agent.Context, laya.State) (OUT, error)) (*workflow.FunctionNode, error) {
	if err := validateNodeOptions(opts); err != nil {
		return nil, err
	}
	if projector == nil {
		return nil, invalidConfig("new node %q: state projector is nil", opts.Name)
	}
	fn := func(ctx agent.Context, input IN) (OUT, error) {
		if err := ctx.Err(); err != nil {
			var zero OUT
			return zero, err
		}
		state, err := projector(ctx, input)
		if err != nil {
			var zero OUT
			return zero, fmt.Errorf("project state: %w", err)
		}
		if err := ctx.Err(); err != nil {
			var zero OUT
			return zero, err
		}
		output, err := run(ctx, state)
		if err != nil {
			var zero OUT
			return zero, err
		}
		if err := ctx.Err(); err != nil {
			var zero OUT
			return zero, err
		}
		return output, nil
	}
	return workflow.NewFunctionNode(opts.Name, fn, cloneNodeConfig(opts.Config)), nil
}

func resultOutput(result laya.Result) (ResultOutput, error) {
	switch result := result.(type) {
	case laya.ChoiceResult:
		answer := laya.ChoiceAnswer{Selected: result.Selected(), Probabilities: result.Probabilities(), Confidence: result.Confidence(), ActionProbability: result.ActionProbability()}
		return ResultOutput{Kind: ResultKindChoice, QuestionID: result.QuestionID(), Choice: &answer}, nil
	case laya.ScoreResult:
		answer := laya.ScoreAnswer{ExpectedLevel: result.ExpectedLevel(), Distribution: result.Distribution(), Confidence: result.Confidence(), ActionProbability: result.ActionProbability()}
		return ResultOutput{Kind: ResultKindScore, QuestionID: result.QuestionID(), Score: &answer}, nil
	case laya.NoulResult:
		answer := laya.NoulAnswer{TrueProbability: result.TrueProbability(), ActionProbability: result.ActionProbability()}
		return ResultOutput{Kind: ResultKindNoul, QuestionID: result.QuestionID(), Noul: &answer}, nil
	default:
		return ResultOutput{}, fmt.Errorf("unsupported result type %T", result)
	}
}

func validateNodeOptions(opts NodeOptions) error {
	if strings.TrimSpace(opts.Name) == "" {
		return invalidConfig("node name is empty")
	}
	if opts.Name == "user" {
		return invalidConfig("node name %q is reserved by ADK", opts.Name)
	}
	return nil
}

func invalidConfig(format string, args ...any) error {
	return fmt.Errorf("%w: %s", laya.ErrInvalidConfig, fmt.Sprintf(format, args...))
}

func cloneNodeConfig(cfg workflow.NodeConfig) workflow.NodeConfig {
	if cfg.RerunOnResume != nil {
		value := *cfg.RerunOnResume
		cfg.RerunOnResume = &value
	}
	if cfg.WaitForOutput != nil {
		value := *cfg.WaitForOutput
		cfg.WaitForOutput = &value
	}
	if cfg.RetryConfig != nil {
		value := *cfg.RetryConfig
		cfg.RetryConfig = &value
	}
	return cfg
}
