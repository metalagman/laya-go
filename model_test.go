package laya

import (
	"context"
	"errors"
	"runtime"
	"testing"
	"time"
)

func TestModelPredictsSharedStateInMixedOrder(t *testing.T) {
	var gotState State
	var gotIDs []QuestionID
	engine := &fakeModelEngine{predict: func(_ context.Context, state State, questions []Question) (Prediction, error) {
		gotState = state
		for _, question := range questions {
			gotIDs = append(gotIDs, question.ID())
		}
		return fakePrediction(questions)
	}}
	runtime, model, err := newFakeRuntime(engine, ModelOptions{})
	if err != nil {
		t.Fatalf("newFakeRuntime() returned unexpected error: %v", err)
	}
	t.Cleanup(func() {
		if err := model.Close(context.Background()); err != nil {
			t.Errorf("model.Close() cleanup = %v, want nil", err)
		}
		if err := runtime.Close(context.Background()); err != nil {
			t.Errorf("runtime.Close() cleanup = %v, want nil", err)
		}
	})

	state, err := JSONState([]byte(`{"customer":"alice"}`))
	if err != nil {
		t.Fatalf("JSONState() returned unexpected error: %v", err)
	}
	choice, err := NewChoiceQuestion("route", "Route.", []ChoiceCriterion{{ID: "safe", Description: "safe"}, {ID: "review", Description: "review"}})
	if err != nil {
		t.Fatalf("NewChoiceQuestion() returned unexpected error: %v", err)
	}
	score, err := NewScoreQuestion("quality", "Score.", []ScoreLevel{{Label: "low", Description: "low"}, {Label: "high", Description: "high"}})
	if err != nil {
		t.Fatalf("NewScoreQuestion() returned unexpected error: %v", err)
	}
	noul, err := NewNoulQuestion("valid", "Valid?", "", "")
	if err != nil {
		t.Fatalf("NewNoulQuestion() returned unexpected error: %v", err)
	}
	questions := []Question{choice, score, noul}
	prediction, err := model.Predict(t.Context(), state, questions)
	if err != nil {
		t.Fatalf("model.Predict() returned unexpected error: %v", err)
	}
	questions[0] = noul

	if got, want := gotState.Kind(), StateKindJSON; got != want {
		t.Errorf("engine state kind = %v, want %v", got, want)
	}
	wantIDs := []QuestionID{"route", "quality", "valid"}
	if len(gotIDs) != len(wantIDs) {
		t.Fatalf("engine question IDs = %v, want %v", gotIDs, wantIDs)
	}
	for i, want := range wantIDs {
		if got := gotIDs[i]; got != want {
			t.Errorf("engine question ID %d = %q, want %q", i, got, want)
		}
		if got := prediction.Results()[i].QuestionID(); got != want {
			t.Errorf("prediction result ID %d = %q, want %q", i, got, want)
		}
	}
	if got, want := prediction.Usage(), (Usage{InputTokens: 11, OutputTokens: 3}); got != want {
		t.Errorf("prediction.Usage() = %+v, want %+v", got, want)
	}
}

func TestModelTypedHelpers(t *testing.T) {
	engine := &fakeModelEngine{predict: func(_ context.Context, _ State, questions []Question) (Prediction, error) {
		return fakePrediction(questions)
	}}
	runtime, model, err := newFakeRuntime(engine, ModelOptions{})
	if err != nil {
		t.Fatalf("newFakeRuntime() returned unexpected error: %v", err)
	}
	t.Cleanup(func() {
		if err := model.Close(context.Background()); err != nil {
			t.Errorf("model.Close() cleanup = %v, want nil", err)
		}
		if err := runtime.Close(context.Background()); err != nil {
			t.Errorf("runtime.Close() cleanup = %v, want nil", err)
		}
	})
	state := TextState("shared")
	choice, err := NewChoiceQuestion("route", "Route.", []ChoiceCriterion{{ID: "safe", Description: "safe"}, {ID: "review", Description: "review"}})
	if err != nil {
		t.Fatalf("NewChoiceQuestion() returned unexpected error: %v", err)
	}
	score, err := NewScoreQuestion("quality", "Score.", []ScoreLevel{{Label: "low", Description: "low"}, {Label: "high", Description: "high"}})
	if err != nil {
		t.Fatalf("NewScoreQuestion() returned unexpected error: %v", err)
	}
	noul, err := NewNoulQuestion("valid", "Valid?", "", "")
	if err != nil {
		t.Fatalf("NewNoulQuestion() returned unexpected error: %v", err)
	}

	choiceResult, err := model.Choice(t.Context(), state, choice)
	if err != nil || choiceResult.QuestionID() != "route" {
		t.Errorf("model.Choice() = (%+v, %v), want question ID route and nil error", choiceResult, err)
	}
	scoreResult, err := model.Score(t.Context(), state, score)
	if err != nil || scoreResult.QuestionID() != "quality" {
		t.Errorf("model.Score() = (%+v, %v), want question ID quality and nil error", scoreResult, err)
	}
	noulResult, err := model.Noul(t.Context(), state, noul)
	if err != nil || noulResult.QuestionID() != "valid" {
		t.Errorf("model.Noul() = (%+v, %v), want question ID valid and nil error", noulResult, err)
	}
}

func TestModelRejectsMismatchedEngineOutput(t *testing.T) {
	choice, err := NewChoiceQuestion("answer", "Choose.", []ChoiceCriterion{
		{ID: "first", Description: "first"},
		{ID: "second", Description: "second"},
	})
	if err != nil {
		t.Fatalf("NewChoiceQuestion() returned unexpected error: %v", err)
	}
	score, err := NewScoreQuestion("answer", "Score.", []ScoreLevel{
		{Label: "low", Description: "low"},
		{Label: "high", Description: "high"},
	})
	if err != nil {
		t.Fatalf("NewScoreQuestion() returned unexpected error: %v", err)
	}
	noul, err := NewNoulQuestion("answer", "True?", "", "")
	if err != nil {
		t.Fatalf("NewNoulQuestion() returned unexpected error: %v", err)
	}

	wrongType, err := NewNoulResult("answer", NoulAnswer{TrueProbability: 0.5})
	if err != nil {
		t.Fatalf("NewNoulResult() returned unexpected error: %v", err)
	}
	wrongChoiceOrder, err := NewChoiceResult("answer", ChoiceAnswer{
		Selected: "first",
		Probabilities: []CriterionProbability{
			{CriterionID: "second", Probability: 0.5},
			{CriterionID: "first", Probability: 0.5},
		},
		Confidence: 0.5,
	})
	if err != nil {
		t.Fatalf("NewChoiceResult() returned unexpected error: %v", err)
	}
	wrongScoreLabels, err := NewScoreResult("answer", ScoreAnswer{
		ExpectedLevel: 0.5,
		Distribution: []ScoreProbability{
			{Level: 0, Label: "bad", Probability: 0.5},
			{Level: 1, Label: "good", Probability: 0.5},
		},
		Confidence: 0.5,
	})
	if err != nil {
		t.Fatalf("NewScoreResult() returned unexpected error: %v", err)
	}

	tests := []struct {
		name     string
		question Question
		result   Result
	}{
		{name: "wrong result type", question: choice, result: wrongType},
		{name: "choice criteria order", question: choice, result: wrongChoiceOrder},
		{name: "score rubric labels", question: score, result: wrongScoreLabels},
		{name: "wrong noul result type", question: noul, result: wrongChoiceOrder},
	}

	var output Prediction
	engine := &fakeModelEngine{predict: func(_ context.Context, _ State, _ []Question) (Prediction, error) {
		return output, nil
	}}
	runtimeOwner, model, err := newFakeRuntime(engine, ModelOptions{})
	if err != nil {
		t.Fatalf("newFakeRuntime() returned unexpected error: %v", err)
	}
	t.Cleanup(func() {
		if err := model.Close(context.Background()); err != nil {
			t.Errorf("model.Close() cleanup = %v, want nil", err)
		}
		if err := runtimeOwner.Close(context.Background()); err != nil {
			t.Errorf("runtime.Close() cleanup = %v, want nil", err)
		}
	})

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			output, err = NewPrediction([]Result{test.result}, Usage{}, PredictionMetadata{})
			if err != nil {
				t.Fatalf("NewPrediction() returned unexpected error: %v", err)
			}
			if _, err := model.Predict(t.Context(), TextState("state"), []Question{test.question}); !errors.Is(err, ErrInvalidOutput) {
				t.Errorf("model.Predict() error = %v, want errors.Is(_, ErrInvalidOutput)", err)
			}
		})
	}
}

func TestModelRejectsRequestsBeforeEngineAdmission(t *testing.T) {
	engine := &fakeModelEngine{predict: func(_ context.Context, _ State, questions []Question) (Prediction, error) {
		return fakePrediction(questions)
	}}
	runtimeOwner, model, err := newFakeRuntime(engine, ModelOptions{})
	if err != nil {
		t.Fatalf("newFakeRuntime() returned unexpected error: %v", err)
	}
	t.Cleanup(func() {
		if err := model.Close(context.Background()); err != nil {
			t.Errorf("model.Close() cleanup = %v, want nil", err)
		}
		if err := runtimeOwner.Close(context.Background()); err != nil {
			t.Errorf("runtime.Close() cleanup = %v, want nil", err)
		}
	})

	question, err := NewNoulQuestion("valid", "Valid?", "", "")
	if err != nil {
		t.Fatalf("NewNoulQuestion() returned unexpected error: %v", err)
	}
	var nilQuestion *NoulQuestion
	canceled, cancel := context.WithCancel(t.Context())
	cancel()
	tests := []struct {
		name      string
		ctx       context.Context
		state     State
		questions []Question
		want      error
	}{
		{name: "nil context", state: TextState("state"), questions: []Question{question}, want: ErrInvalidRequest},
		{name: "canceled context", ctx: canceled, state: TextState("state"), questions: []Question{question}, want: context.Canceled},
		{name: "zero state", ctx: t.Context(), questions: []Question{question}, want: ErrInvalidState},
		{name: "empty questions", ctx: t.Context(), state: TextState("state"), want: ErrInvalidRequest},
		{name: "duplicate question IDs", ctx: t.Context(), state: TextState("state"), questions: []Question{question, question}, want: ErrInvalidRequest},
		{name: "typed nil question", ctx: t.Context(), state: TextState("state"), questions: []Question{nilQuestion}, want: ErrInvalidRequest},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := model.Predict(test.ctx, test.state, test.questions); !errors.Is(err, test.want) {
				t.Errorf("model.Predict() error = %v, want errors.Is(_, %v)", err, test.want)
			}
		})
	}
	predictCalls, _ := engine.counts()
	if predictCalls != 0 {
		t.Errorf("model engine predict calls = %d, want 0", predictCalls)
	}
}

func TestModelRejectsMalformedPredictionShape(t *testing.T) {
	question, err := NewNoulQuestion("expected", "Valid?", "", "")
	if err != nil {
		t.Fatalf("NewNoulQuestion() returned unexpected error: %v", err)
	}
	expected, err := NewNoulResult("expected", NoulAnswer{TrueProbability: 0.5})
	if err != nil {
		t.Fatalf("NewNoulResult(expected) returned unexpected error: %v", err)
	}
	other, err := NewNoulResult("other", NoulAnswer{TrueProbability: 0.5})
	if err != nil {
		t.Fatalf("NewNoulResult(other) returned unexpected error: %v", err)
	}
	tests := []struct {
		name    string
		results []Result
	}{
		{name: "wrong result count", results: []Result{expected, other}},
		{name: "wrong question ID", results: []Result{other}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			prediction, err := NewPrediction(test.results, Usage{}, PredictionMetadata{})
			if err != nil {
				t.Fatalf("NewPrediction() returned unexpected error: %v", err)
			}
			engine := &fakeModelEngine{predict: func(context.Context, State, []Question) (Prediction, error) {
				return prediction, nil
			}}
			runtimeOwner, model, err := newFakeRuntime(engine, ModelOptions{})
			if err != nil {
				t.Fatalf("newFakeRuntime() returned unexpected error: %v", err)
			}
			defer func() {
				if err := model.Close(context.Background()); err != nil {
					t.Errorf("model.Close() cleanup = %v, want nil", err)
				}
				if err := runtimeOwner.Close(context.Background()); err != nil {
					t.Errorf("runtime.Close() cleanup = %v, want nil", err)
				}
			}()

			if _, err := model.Predict(t.Context(), TextState("state"), []Question{question}); !errors.Is(err, ErrInvalidOutput) {
				t.Errorf("model.Predict() error = %v, want errors.Is(_, ErrInvalidOutput)", err)
			}
		})
	}
}

func TestModelPreservesVisibleTruncationMetadata(t *testing.T) {
	question, err := NewNoulQuestion("valid", "Valid?", "", "")
	if err != nil {
		t.Fatalf("NewNoulQuestion() returned unexpected error: %v", err)
	}
	result, err := NewNoulResult("valid", NoulAnswer{TrueProbability: 0.5})
	if err != nil {
		t.Fatalf("NewNoulResult() returned unexpected error: %v", err)
	}
	want := TruncationMetadata{Truncated: true, OriginalInputTokens: 10, EffectiveInputTokens: 4}
	prediction, err := NewPrediction([]Result{result}, Usage{InputTokens: 4, OutputTokens: 1}, PredictionMetadata{Truncation: want})
	if err != nil {
		t.Fatalf("NewPrediction() returned unexpected error: %v", err)
	}
	engine := &fakeModelEngine{predict: func(context.Context, State, []Question) (Prediction, error) {
		return prediction, nil
	}}
	runtimeOwner, model, err := newFakeRuntime(engine, ModelOptions{InputTokenLimit: 4, Truncation: TruncateOverflow})
	if err != nil {
		t.Fatalf("newFakeRuntime() returned unexpected error: %v", err)
	}
	defer func() {
		if err := model.Close(context.Background()); err != nil {
			t.Errorf("model.Close() cleanup = %v, want nil", err)
		}
		if err := runtimeOwner.Close(context.Background()); err != nil {
			t.Errorf("runtime.Close() cleanup = %v, want nil", err)
		}
	}()

	got, err := model.Predict(t.Context(), TextState("long state"), []Question{question})
	if err != nil {
		t.Fatalf("model.Predict() returned unexpected error: %v", err)
	}
	if got.Metadata().Truncation != want {
		t.Errorf("model.Predict().Metadata().Truncation = %+v, want %+v", got.Metadata().Truncation, want)
	}
}

func TestModelAdmissionIsBoundedAndFIFO(t *testing.T) {
	started := make(chan QuestionID, 3)
	release := make(chan struct{}, 3)
	engine := &fakeModelEngine{predict: func(ctx context.Context, _ State, questions []Question) (Prediction, error) {
		started <- questions[0].ID()
		select {
		case <-release:
			return fakePrediction(questions)
		case <-ctx.Done():
			return Prediction{}, ctx.Err()
		}
	}}
	runtimeOwner, model, err := newFakeRuntime(engine, ModelOptions{QueueCapacity: 2})
	if err != nil {
		t.Fatalf("newFakeRuntime() returned unexpected error: %v", err)
	}
	t.Cleanup(func() {
		if err := model.Close(context.Background()); err != nil {
			t.Errorf("model.Close() cleanup = %v, want nil", err)
		}
		if err := runtimeOwner.Close(context.Background()); err != nil {
			t.Errorf("runtime.Close() cleanup = %v, want nil", err)
		}
	})

	state := TextState("shared")
	questions := make([]NoulQuestion, 4)
	for i, id := range []QuestionID{"first", "second", "third", "overflow"} {
		questions[i], err = NewNoulQuestion(id, "Valid?", "", "")
		if err != nil {
			t.Fatalf("NewNoulQuestion(%q) returned unexpected error: %v", id, err)
		}
	}
	type outcome struct {
		id  QuestionID
		err error
	}
	outcomes := make(chan outcome, 3)
	go func() {
		_, err := model.Noul(t.Context(), state, questions[0])
		outcomes <- outcome{id: "first", err: err}
	}()
	if got := <-started; got != "first" {
		t.Fatalf("first started question = %q, want first", got)
	}

	for i := 1; i <= 2; i++ {
		question := questions[i]
		go func() {
			_, err := model.Noul(t.Context(), state, question)
			outcomes <- outcome{id: question.ID(), err: err}
		}()
		waitForQueued(t, model, i)
	}
	if _, err := model.Noul(t.Context(), state, questions[3]); !errors.Is(err, ErrQueueFull) {
		t.Errorf("overflow model.Noul() error = %v, want errors.Is(_, ErrQueueFull)", err)
	}

	for _, want := range []QuestionID{"second", "third"} {
		release <- struct{}{}
		if got := <-started; got != want {
			t.Errorf("next started question = %q, want %q", got, want)
		}
	}
	release <- struct{}{}
	for range 3 {
		outcome := <-outcomes
		if outcome.err != nil {
			t.Errorf("model.Noul(%q) error = %v, want nil", outcome.id, outcome.err)
		}
	}
}

func TestModelQueueTimeout(t *testing.T) {
	started := make(chan struct{}, 1)
	release := make(chan struct{})
	engine := &fakeModelEngine{predict: func(ctx context.Context, _ State, questions []Question) (Prediction, error) {
		started <- struct{}{}
		select {
		case <-release:
			return fakePrediction(questions)
		case <-ctx.Done():
			return Prediction{}, ctx.Err()
		}
	}}
	runtimeOwner, model, err := newFakeRuntime(engine, ModelOptions{QueueCapacity: 1, MaxQueueWait: time.Nanosecond})
	if err != nil {
		t.Fatalf("newFakeRuntime() returned unexpected error: %v", err)
	}
	state := TextState("shared")
	question, err := NewNoulQuestion("valid", "Valid?", "", "")
	if err != nil {
		t.Fatalf("NewNoulQuestion() returned unexpected error: %v", err)
	}
	activeDone := make(chan error, 1)
	go func() {
		_, err := model.Noul(t.Context(), state, question)
		activeDone <- err
	}()
	<-started

	if _, err := model.Noul(t.Context(), state, question); !errors.Is(err, ErrQueueTimeout) {
		t.Errorf("queued model.Noul() error = %v, want errors.Is(_, ErrQueueTimeout)", err)
	}

	close(release)
	if err := <-activeDone; err != nil {
		t.Errorf("active model.Noul() error = %v, want nil", err)
	}
	if err := model.Close(t.Context()); err != nil {
		t.Fatalf("model.Close() returned unexpected error: %v", err)
	}
	if err := runtimeOwner.Close(t.Context()); err != nil {
		t.Fatalf("runtime.Close() returned unexpected error: %v", err)
	}
}

func TestModelQueuedCancellation(t *testing.T) {
	started := make(chan struct{}, 1)
	release := make(chan struct{})
	engine := &fakeModelEngine{predict: func(ctx context.Context, _ State, questions []Question) (Prediction, error) {
		started <- struct{}{}
		select {
		case <-release:
			return fakePrediction(questions)
		case <-ctx.Done():
			return Prediction{}, ctx.Err()
		}
	}}
	runtimeOwner, model, err := newFakeRuntime(engine, ModelOptions{QueueCapacity: 1})
	if err != nil {
		t.Fatalf("newFakeRuntime() returned unexpected error: %v", err)
	}
	state := TextState("shared")
	question, err := NewNoulQuestion("valid", "Valid?", "", "")
	if err != nil {
		t.Fatalf("NewNoulQuestion() returned unexpected error: %v", err)
	}
	activeDone := make(chan error, 1)
	go func() {
		_, err := model.Noul(t.Context(), state, question)
		activeDone <- err
	}()
	<-started

	cancelCtx, cancel := context.WithCancel(t.Context())
	queuedDone := make(chan error, 1)
	go func() {
		_, err := model.Noul(cancelCtx, state, question)
		queuedDone <- err
	}()
	waitForQueued(t, model, 1)
	cancel()
	if err := <-queuedDone; !errors.Is(err, context.Canceled) {
		t.Errorf("canceled queued model.Noul() error = %v, want errors.Is(_, context.Canceled)", err)
	}
	close(release)
	if err := <-activeDone; err != nil {
		t.Errorf("active model.Noul() error = %v, want nil", err)
	}
	if err := model.Close(t.Context()); err != nil {
		t.Fatalf("model.Close() returned unexpected error: %v", err)
	}
	if err := runtimeOwner.Close(t.Context()); err != nil {
		t.Fatalf("runtime.Close() returned unexpected error: %v", err)
	}
}

func TestModelCloseRetainsResourcesUntilInferenceFinishes(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	engine := &fakeModelEngine{predict: func(_ context.Context, _ State, questions []Question) (Prediction, error) {
		close(started)
		<-release
		return fakePrediction(questions)
	}}
	runtimeOwner, model, err := newFakeRuntime(engine, ModelOptions{})
	if err != nil {
		t.Fatalf("newFakeRuntime() returned unexpected error: %v", err)
	}
	question, err := NewNoulQuestion("valid", "Valid?", "", "")
	if err != nil {
		t.Fatalf("NewNoulQuestion() returned unexpected error: %v", err)
	}
	predictDone := make(chan error, 1)
	go func() {
		_, err := model.Noul(t.Context(), TextState("shared"), question)
		predictDone <- err
	}()
	<-started

	closeCtx, cancelClose := context.WithCancel(t.Context())
	closeDone := make(chan error, 1)
	go func() { closeDone <- model.Close(closeCtx) }()
	<-model.closing
	cancelClose()
	if err := <-closeDone; !errors.Is(err, context.Canceled) {
		t.Errorf("model.Close(canceled) error = %v, want errors.Is(_, context.Canceled)", err)
	}
	if _, err := model.Noul(t.Context(), TextState("shared"), question); !errors.Is(err, ErrModelClosed) {
		t.Errorf("model.Noul() while closing error = %v, want errors.Is(_, ErrModelClosed)", err)
	}
	_, closeCalls := engine.counts()
	if closeCalls != 0 {
		t.Errorf("engine close calls before inference release = %d, want 0", closeCalls)
	}

	close(release)
	if err := <-predictDone; err != nil {
		t.Errorf("in-flight model.Noul() error = %v, want nil", err)
	}
	if err := model.Close(t.Context()); err != nil {
		t.Fatalf("retry model.Close() returned unexpected error: %v", err)
	}
	if err := runtimeOwner.Close(t.Context()); err != nil {
		t.Fatalf("runtime.Close() returned unexpected error: %v", err)
	}
}

func TestModelCloseDrainsAlreadyQueuedInference(t *testing.T) {
	started := make(chan QuestionID, 2)
	release := make(chan struct{}, 2)
	engine := &fakeModelEngine{predict: func(_ context.Context, _ State, questions []Question) (Prediction, error) {
		started <- questions[0].ID()
		<-release
		return fakePrediction(questions)
	}}
	runtimeOwner, model, err := newFakeRuntime(engine, ModelOptions{QueueCapacity: 1})
	if err != nil {
		t.Fatalf("newFakeRuntime() returned unexpected error: %v", err)
	}
	first, err := NewNoulQuestion("first", "Valid?", "", "")
	if err != nil {
		t.Fatalf("NewNoulQuestion(first) returned unexpected error: %v", err)
	}
	second, err := NewNoulQuestion("second", "Valid?", "", "")
	if err != nil {
		t.Fatalf("NewNoulQuestion(second) returned unexpected error: %v", err)
	}
	late, err := NewNoulQuestion("late", "Valid?", "", "")
	if err != nil {
		t.Fatalf("NewNoulQuestion(late) returned unexpected error: %v", err)
	}

	predictDone := make(chan error, 2)
	go func() {
		_, err := model.Noul(t.Context(), TextState("state"), first)
		predictDone <- err
	}()
	if got := <-started; got != "first" {
		t.Fatalf("first started question = %q, want first", got)
	}
	go func() {
		_, err := model.Noul(t.Context(), TextState("state"), second)
		predictDone <- err
	}()
	waitForQueued(t, model, 1)

	closeDone := make(chan error, 1)
	go func() { closeDone <- model.Close(t.Context()) }()
	<-model.closing
	if _, err := model.Noul(t.Context(), TextState("state"), late); !errors.Is(err, ErrModelClosed) {
		t.Errorf("late model.Noul() error = %v, want errors.Is(_, ErrModelClosed)", err)
	}
	release <- struct{}{}
	if got := <-started; got != "second" {
		t.Errorf("second started question = %q, want second", got)
	}
	release <- struct{}{}
	for range 2 {
		if err := <-predictDone; err != nil {
			t.Errorf("admitted model.Noul() error = %v, want nil", err)
		}
	}
	if err := <-closeDone; err != nil {
		t.Errorf("model.Close() error = %v, want nil", err)
	}
	if err := runtimeOwner.Close(t.Context()); err != nil {
		t.Fatalf("runtime.Close() returned unexpected error: %v", err)
	}
}

func TestModelCloseCanRetryAfterEngineError(t *testing.T) {
	closeAttempt := 0
	engine := &fakeModelEngine{
		predict: func(_ context.Context, _ State, questions []Question) (Prediction, error) {
			return fakePrediction(questions)
		},
		close: func(context.Context) error {
			closeAttempt++
			if closeAttempt == 1 {
				return context.DeadlineExceeded
			}
			return nil
		},
	}
	runtimeOwner, model, err := newFakeRuntime(engine, ModelOptions{})
	if err != nil {
		t.Fatalf("newFakeRuntime() returned unexpected error: %v", err)
	}

	if err := model.Close(t.Context()); !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("first model.Close() error = %v, want errors.Is(_, context.DeadlineExceeded)", err)
	}
	if err := runtimeOwner.Close(t.Context()); !errors.Is(err, ErrModelsOpen) {
		t.Errorf("runtime.Close() after failed model close = %v, want errors.Is(_, ErrModelsOpen)", err)
	}
	if err := model.Close(t.Context()); err != nil {
		t.Fatalf("second model.Close() returned unexpected error: %v", err)
	}
	if err := runtimeOwner.Close(t.Context()); err != nil {
		t.Fatalf("runtime.Close() returned unexpected error: %v", err)
	}
	_, closeCalls := engine.counts()
	if closeCalls != 2 {
		t.Errorf("model engine close calls = %d, want 2", closeCalls)
	}
}

func TestModelPreservesBudgetAndContextErrors(t *testing.T) {
	started := make(chan struct{})
	engine := &fakeModelEngine{predict: func(ctx context.Context, state State, _ []Question) (Prediction, error) {
		text, _ := state.Text()
		if len(text) > 3 {
			return Prediction{}, &InputTooLongError{InputTokens: len(text), Limit: 3}
		}
		close(started)
		<-ctx.Done()
		return Prediction{}, ctx.Err()
	}}
	runtimeOwner, model, err := newFakeRuntime(engine, ModelOptions{InputTokenLimit: 3})
	if err != nil {
		t.Fatalf("newFakeRuntime() returned unexpected error: %v", err)
	}
	question, err := NewNoulQuestion("valid", "Valid?", "", "")
	if err != nil {
		t.Fatalf("NewNoulQuestion() returned unexpected error: %v", err)
	}
	if _, err := model.Noul(t.Context(), TextState("too long"), question); !errors.Is(err, ErrInputTooLong) {
		t.Errorf("model.Noul(long state) error = %v, want errors.Is(_, ErrInputTooLong)", err)
	}

	ctx, cancel := context.WithCancel(t.Context())
	done := make(chan error, 1)
	go func() {
		_, err := model.Noul(ctx, TextState("ok"), question)
		done <- err
	}()
	<-started
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Errorf("model.Noul(canceled) error = %v, want errors.Is(_, context.Canceled)", err)
	}
	if err := model.Close(t.Context()); err != nil {
		t.Fatalf("model.Close() returned unexpected error: %v", err)
	}
	if err := runtimeOwner.Close(t.Context()); err != nil {
		t.Fatalf("runtime.Close() returned unexpected error: %v", err)
	}
}

func TestModelConcurrentCloseRunsEngineCleanupOnce(t *testing.T) {
	started := make(chan struct{}, 1)
	release := make(chan struct{})
	engine := &fakeModelEngine{
		predict: func(_ context.Context, _ State, questions []Question) (Prediction, error) {
			return fakePrediction(questions)
		},
		close: func(ctx context.Context) error {
			started <- struct{}{}
			select {
			case <-release:
				return nil
			case <-ctx.Done():
				return ctx.Err()
			}
		},
	}
	runtimeOwner, model, err := newFakeRuntime(engine, ModelOptions{})
	if err != nil {
		t.Fatalf("newFakeRuntime() returned unexpected error: %v", err)
	}

	done := make(chan error, 2)
	go func() { done <- model.Close(t.Context()) }()
	<-started
	go func() { done <- model.Close(t.Context()) }()
	close(release)
	for range 2 {
		if err := <-done; err != nil {
			t.Errorf("concurrent model.Close() = %v, want nil", err)
		}
	}
	_, closeCalls := engine.counts()
	if closeCalls != 1 {
		t.Errorf("model engine close calls = %d, want 1", closeCalls)
	}
	if err := runtimeOwner.Close(t.Context()); err != nil {
		t.Fatalf("runtime.Close() returned unexpected error: %v", err)
	}
	canceled, cancel := context.WithCancel(t.Context())
	cancel()
	if err := model.Close(canceled); err != nil {
		t.Errorf("closed model.Close(canceled) = %v, want nil", err)
	}
}

func TestModelZeroValueFailsClosed(t *testing.T) {
	var model Model
	if err := model.Close(t.Context()); !errors.Is(err, ErrInvalidState) {
		t.Errorf("(Model{}).Close() error = %v, want errors.Is(_, ErrInvalidState)", err)
	}
	question, err := NewNoulQuestion("valid", "Valid?", "", "")
	if err != nil {
		t.Fatalf("NewNoulQuestion() returned unexpected error: %v", err)
	}
	if _, err := model.Noul(t.Context(), TextState("state"), question); !errors.Is(err, ErrInvalidState) {
		t.Errorf("(Model{}).Noul() error = %v, want errors.Is(_, ErrInvalidState)", err)
	}
}

func waitForQueued(t *testing.T, model *Model, want int) {
	t.Helper()
	for {
		model.mu.Lock()
		got := len(model.waiters)
		model.mu.Unlock()
		if got == want {
			return
		}
		select {
		case <-t.Context().Done():
			t.Fatalf("queued callers = %d, want %d before test context ended", got, want)
		default:
			runtime.Gosched()
		}
	}
}
