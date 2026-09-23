package laya

import (
	"context"
	"fmt"
	"io/fs"
	"sync"
)

type fakeRuntimeEngine struct {
	mu sync.Mutex

	dirModel modelEngine
	fsModel  modelEngine
	dirOpen  func(context.Context, string, ModelOptions) (modelEngine, error)
	fsOpen   func(context.Context, fs.FS, FSModelOptions) (modelEngine, error)
	close    func(context.Context) error

	dirCalls   int
	fsCalls    int
	closeCalls int
}

func (e *fakeRuntimeEngine) OpenModelDir(ctx context.Context, dir string, opts ModelOptions) (modelEngine, error) {
	e.mu.Lock()
	e.dirCalls++
	open := e.dirOpen
	model := e.dirModel
	e.mu.Unlock()
	if open != nil {
		return open(ctx, dir, opts)
	}
	return model, nil
}

func (e *fakeRuntimeEngine) OpenModelFS(ctx context.Context, fsys fs.FS, opts FSModelOptions) (modelEngine, error) {
	e.mu.Lock()
	e.fsCalls++
	open := e.fsOpen
	model := e.fsModel
	e.mu.Unlock()
	if open != nil {
		return open(ctx, fsys, opts)
	}
	return model, nil
}

func (e *fakeRuntimeEngine) Close(ctx context.Context) error {
	e.mu.Lock()
	e.closeCalls++
	closeRuntime := e.close
	e.mu.Unlock()
	if closeRuntime != nil {
		return closeRuntime(ctx)
	}
	return nil
}

func (e *fakeRuntimeEngine) counts() (dir, fsys, close int) {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.dirCalls, e.fsCalls, e.closeCalls
}

type fakeModelEngine struct {
	mu sync.Mutex

	predict func(context.Context, State, []Question) (Prediction, error)
	close   func(context.Context) error

	predictCalls int
	closeCalls   int
}

func (e *fakeModelEngine) Predict(ctx context.Context, state State, questions []Question) (Prediction, error) {
	e.mu.Lock()
	e.predictCalls++
	predict := e.predict
	e.mu.Unlock()
	if predict == nil {
		return Prediction{}, ErrNativeFailure
	}
	return predict(ctx, state, questions)
}

func (e *fakeModelEngine) Close(ctx context.Context) error {
	e.mu.Lock()
	e.closeCalls++
	closeModel := e.close
	e.mu.Unlock()
	if closeModel != nil {
		return closeModel(ctx)
	}
	return nil
}

func (e *fakeModelEngine) counts() (predict, close int) {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.predictCalls, e.closeCalls
}

func fakePrediction(questions []Question) (Prediction, error) {
	results := make([]Result, 0, len(questions))
	for _, question := range questions {
		switch question := question.(type) {
		case ChoiceQuestion:
			criteria := question.Criteria()
			probabilities := make([]CriterionProbability, len(criteria))
			for i, criterion := range criteria {
				probabilities[i] = CriterionProbability{CriterionID: criterion.ID, Probability: 1 / float64(len(criteria))}
			}
			result, err := NewChoiceResult(question.ID(), ChoiceAnswer{
				Selected:          criteria[0].ID,
				Probabilities:     probabilities,
				Confidence:        0.5,
				ActionProbability: 0.25,
			})
			if err != nil {
				return Prediction{}, fmt.Errorf("build choice fixture: %w", err)
			}
			results = append(results, result)
		case ScoreQuestion:
			rubric := question.Rubric()
			distribution := make([]ScoreProbability, len(rubric))
			for i, level := range rubric {
				distribution[i] = ScoreProbability{Level: i, Label: level.Label, Probability: 1 / float64(len(rubric))}
			}
			result, err := NewScoreResult(question.ID(), ScoreAnswer{
				ExpectedLevel:     float64(len(rubric)-1) / 2,
				Distribution:      distribution,
				Confidence:        0.5,
				ActionProbability: 0.25,
			})
			if err != nil {
				return Prediction{}, fmt.Errorf("build score fixture: %w", err)
			}
			results = append(results, result)
		case NoulQuestion:
			result, err := NewNoulResult(question.ID(), NoulAnswer{
				TrueProbability:   0.75,
				ActionProbability: 0.25,
			})
			if err != nil {
				return Prediction{}, fmt.Errorf("build noul fixture: %w", err)
			}
			results = append(results, result)
		default:
			return Prediction{}, fmt.Errorf("unsupported fixture question %T", question)
		}
	}
	return NewPrediction(results, Usage{InputTokens: 11, OutputTokens: len(results)}, PredictionMetadata{})
}

func newFakeRuntime(model *fakeModelEngine, opts ModelOptions) (*Runtime, *Model, error) {
	engine := &fakeRuntimeEngine{dirOpen: func(_ context.Context, _ string, got ModelOptions) (modelEngine, error) {
		if got != opts {
			return nil, fmt.Errorf("model options = %+v, want %+v", got, opts)
		}
		return model, nil
	}}
	runtime, err := newRuntime(engine)
	if err != nil {
		return nil, nil, err
	}
	opened, err := runtime.OpenModelDir(context.Background(), "/fake/model", opts)
	if err != nil {
		return nil, nil, err
	}
	return runtime, opened, nil
}
