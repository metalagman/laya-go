package laya

import (
	"context"
	"fmt"
	"sync"
	"time"
)

type modelState uint8

const (
	modelOpen modelState = iota + 1
	modelClosing
	modelClosed
)

type admissionWaiter struct {
	ready   chan struct{}
	granted bool
}

// Model is a shareable handle to one opened Laya model.
//
// Model inference methods are safe for simultaneous use. Admission is bounded
// by ModelOptions: at most one call is active in this foundation contract and
// QueueCapacity callers may wait in FIFO order. Callers own the Model and must
// call Close before closing its Runtime.
type Model struct {
	mu sync.Mutex

	runtime *Runtime
	engine  modelEngine
	opts    ModelOptions
	state   modelState

	active  bool
	waiters []*admissionWaiter
	idle    chan struct{}
	idleSet bool
	closing chan struct{}

	closeInProgress bool
	closeDone       chan struct{}
}

// Predict evaluates one immutable state against questions in input order.
//
// Predict returns one Prediction whose results preserve question order and
// whose Usage applies to the complete batch. Mutable input slices are copied
// before admission. Cancellation while queued or running returns an error
// matching the caller context; native cleanup remains owned by the engine.
func (m *Model) Predict(ctx context.Context, state State, questions []Question) (Prediction, error) {
	if err := validateContext(ctx); err != nil {
		return Prediction{}, fmt.Errorf("predict: %w", err)
	}
	if err := state.validate(); err != nil {
		return Prediction{}, fmt.Errorf("predict: %w", err)
	}
	questions = append([]Question(nil), questions...)
	if err := validateQuestions(questions); err != nil {
		return Prediction{}, fmt.Errorf("predict: %w", err)
	}
	if err := m.acquire(ctx); err != nil {
		return Prediction{}, fmt.Errorf("predict: %w", err)
	}
	defer m.release()

	prediction, err := m.engine.Predict(ctx, state, questions)
	if err != nil {
		return Prediction{}, fmt.Errorf("predict: %w", err)
	}
	if err := validatePredictionForQuestions(prediction, questions); err != nil {
		return Prediction{}, fmt.Errorf("predict: %w", err)
	}
	return prediction, nil
}

// Choice evaluates state against one choice question.
func (m *Model) Choice(ctx context.Context, state State, question ChoiceQuestion) (ChoiceResult, error) {
	prediction, err := m.Predict(ctx, state, []Question{question})
	if err != nil {
		return ChoiceResult{}, err
	}
	return choiceResult(prediction.Results()[0])
}

// Score evaluates state against one ordinal score question.
func (m *Model) Score(ctx context.Context, state State, question ScoreQuestion) (ScoreResult, error) {
	prediction, err := m.Predict(ctx, state, []Question{question})
	if err != nil {
		return ScoreResult{}, err
	}
	return scoreResult(prediction.Results()[0])
}

// Noul evaluates state against one boolean question.
func (m *Model) Noul(ctx context.Context, state State, question NoulQuestion) (NoulResult, error) {
	prediction, err := m.Predict(ctx, state, []Question{question})
	if err != nil {
		return NoulResult{}, err
	}
	return noulResult(prediction.Results()[0])
}

// Close stops new admission, waits for admitted calls, and releases resources.
//
// Close is safe for concurrent use and is idempotent after success. If ctx ends
// before admitted calls or engine cleanup finish, Close returns an error
// matching ctx.Err and leaves the Model in Closing; a later Close may wait or
// retry. The Runtime continues to own the Model until cleanup succeeds.
func (m *Model) Close(ctx context.Context) error {
	if ctx == nil {
		return fmt.Errorf("close model: %w: context is nil", ErrInvalidRequest)
	}
	if m == nil {
		return fmt.Errorf("close model: %w: model is nil", ErrInvalidState)
	}

	for {
		m.mu.Lock()
		switch m.state {
		case modelClosed:
			m.mu.Unlock()
			return nil
		case modelOpen:
			if err := ctx.Err(); err != nil {
				m.mu.Unlock()
				return fmt.Errorf("close model: %w", err)
			}
			m.state = modelClosing
			close(m.closing)
			m.signalIdleLocked()
		case modelClosing:
			if err := ctx.Err(); err != nil {
				m.mu.Unlock()
				return fmt.Errorf("close model: %w", err)
			}
		default:
			m.mu.Unlock()
			return fmt.Errorf("close model: %w: model is not initialized", ErrInvalidState)
		}

		if m.active || len(m.waiters) > 0 {
			idle := m.idle
			m.mu.Unlock()
			select {
			case <-idle:
				continue
			case <-ctx.Done():
				return fmt.Errorf("close model: %w", ctx.Err())
			}
		}

		if m.closeInProgress {
			done := m.closeDone
			m.mu.Unlock()
			select {
			case <-done:
				continue
			case <-ctx.Done():
				return fmt.Errorf("close model: %w", ctx.Err())
			}
		}

		m.closeInProgress = true
		m.closeDone = make(chan struct{})
		done := m.closeDone
		engine := m.engine
		owner := m.runtime
		m.mu.Unlock()

		err := engine.Close(ctx)
		if err == nil {
			owner.unregister(m)
		}

		m.mu.Lock()
		m.closeInProgress = false
		if err == nil {
			m.state = modelClosed
		}
		close(done)
		m.mu.Unlock()

		if err != nil {
			return fmt.Errorf("close model: %w", err)
		}
		return nil
	}
}

func newModel(runtime *Runtime, engine modelEngine, opts ModelOptions) *Model {
	return &Model{
		runtime: runtime,
		engine:  engine,
		opts:    opts,
		state:   modelOpen,
		idle:    make(chan struct{}),
		closing: make(chan struct{}),
	}
}

func (m *Model) acquire(ctx context.Context) error {
	if m == nil {
		return fmt.Errorf("%w: model is nil", ErrInvalidState)
	}
	m.mu.Lock()
	if m.state == modelClosing || m.state == modelClosed {
		m.mu.Unlock()
		return ErrModelClosed
	}
	if m.state != modelOpen {
		m.mu.Unlock()
		return fmt.Errorf("%w: model is not initialized", ErrInvalidState)
	}
	if !m.active {
		m.active = true
		m.mu.Unlock()
		return nil
	}
	if len(m.waiters) >= m.opts.QueueCapacity {
		capacity := m.opts.QueueCapacity
		m.mu.Unlock()
		return &QueueError{Kind: QueueFull, Capacity: capacity}
	}
	waiter := &admissionWaiter{ready: make(chan struct{})}
	m.waiters = append(m.waiters, waiter)
	wait := m.opts.MaxQueueWait
	capacity := m.opts.QueueCapacity
	m.mu.Unlock()

	if wait == 0 {
		select {
		case <-waiter.ready:
			return nil
		case <-ctx.Done():
			m.cancelWaiter(waiter)
			return ctx.Err()
		}
	}

	timer := time.NewTimer(wait)
	defer timer.Stop()
	select {
	case <-waiter.ready:
		return nil
	case <-ctx.Done():
		m.cancelWaiter(waiter)
		return ctx.Err()
	case <-timer.C:
		m.cancelWaiter(waiter)
		return &QueueError{Kind: QueueWaitTimeout, Capacity: capacity, Wait: wait}
	}
}

func (m *Model) cancelWaiter(waiter *admissionWaiter) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if waiter.granted {
		m.releaseLocked()
		return
	}
	for i, candidate := range m.waiters {
		if candidate == waiter {
			copy(m.waiters[i:], m.waiters[i+1:])
			m.waiters[len(m.waiters)-1] = nil
			m.waiters = m.waiters[:len(m.waiters)-1]
			break
		}
	}
	m.signalIdleLocked()
}

func (m *Model) release() {
	m.mu.Lock()
	m.releaseLocked()
	m.mu.Unlock()
}

func (m *Model) releaseLocked() {
	if len(m.waiters) > 0 {
		waiter := m.waiters[0]
		copy(m.waiters, m.waiters[1:])
		m.waiters[len(m.waiters)-1] = nil
		m.waiters = m.waiters[:len(m.waiters)-1]
		waiter.granted = true
		close(waiter.ready)
		return
	}
	m.active = false
	m.signalIdleLocked()
}

func (m *Model) signalIdleLocked() {
	if m.state == modelClosing && !m.active && len(m.waiters) == 0 && !m.idleSet {
		m.idleSet = true
		close(m.idle)
	}
}

func validateQuestions(questions []Question) error {
	if len(questions) == 0 {
		return fmt.Errorf("%w: questions are empty", ErrInvalidRequest)
	}
	seen := make(map[QuestionID]bool, len(questions))
	for i, question := range questions {
		if err := validateQuestion(question); err != nil {
			return fmt.Errorf("question %d: %w", i, err)
		}
		id := question.ID()
		if seen[id] {
			return fmt.Errorf("%w: repeated question ID %q", ErrInvalidRequest, id)
		}
		seen[id] = true
	}
	return nil
}

func validatePredictionForQuestions(prediction Prediction, questions []Question) error {
	if err := prediction.validate(); err != nil {
		return err
	}
	results := prediction.Results()
	if len(results) != len(questions) {
		return fmt.Errorf("%w: got %d results for %d questions", ErrInvalidOutput, len(results), len(questions))
	}
	for i, result := range results {
		if got, want := result.QuestionID(), questions[i].ID(); got != want {
			return fmt.Errorf("%w: result %d question ID %q does not match %q", ErrInvalidOutput, i, got, want)
		}
		if err := validateResultForQuestion(result, questions[i]); err != nil {
			return fmt.Errorf("result %d: %w", i, err)
		}
	}
	return nil
}

func validateResultForQuestion(result Result, question Question) error {
	switch question := question.(type) {
	case ChoiceQuestion:
		return validateChoiceResultForQuestion(result, question)
	case *ChoiceQuestion:
		return validateChoiceResultForQuestion(result, *question)
	case ScoreQuestion:
		return validateScoreResultForQuestion(result, question)
	case *ScoreQuestion:
		return validateScoreResultForQuestion(result, *question)
	case NoulQuestion, *NoulQuestion:
		_, err := noulResult(result)
		return err
	default:
		return fmt.Errorf("%w: unsupported question type %T", ErrInvalidRequest, question)
	}
}

func validateChoiceResultForQuestion(result Result, question ChoiceQuestion) error {
	choice, err := choiceResult(result)
	if err != nil {
		return err
	}
	probabilities := choice.Probabilities()
	criteria := question.Criteria()
	if len(probabilities) != len(criteria) {
		return fmt.Errorf("%w: choice result %q has %d probabilities for %d criteria", ErrInvalidOutput, question.ID(), len(probabilities), len(criteria))
	}
	for i, criterion := range criteria {
		if got := probabilities[i].CriterionID; got != criterion.ID {
			return fmt.Errorf("%w: choice result %q probability %d criterion %q does not match %q", ErrInvalidOutput, question.ID(), i, got, criterion.ID)
		}
	}
	return nil
}

func validateScoreResultForQuestion(result Result, question ScoreQuestion) error {
	score, err := scoreResult(result)
	if err != nil {
		return err
	}
	distribution := score.Distribution()
	rubric := question.Rubric()
	if len(distribution) != len(rubric) {
		return fmt.Errorf("%w: score result %q has %d probabilities for %d levels", ErrInvalidOutput, question.ID(), len(distribution), len(rubric))
	}
	for i, level := range rubric {
		if got := distribution[i].Label; got != level.Label {
			return fmt.Errorf("%w: score result %q level %d label %q does not match %q", ErrInvalidOutput, question.ID(), i, got, level.Label)
		}
	}
	return nil
}

func choiceResult(result Result) (ChoiceResult, error) {
	switch result := result.(type) {
	case ChoiceResult:
		return result, nil
	case *ChoiceResult:
		if result != nil {
			return *result, nil
		}
	}
	return ChoiceResult{}, fmt.Errorf("%w: choice call returned %T", ErrInvalidOutput, result)
}

func scoreResult(result Result) (ScoreResult, error) {
	switch result := result.(type) {
	case ScoreResult:
		return result, nil
	case *ScoreResult:
		if result != nil {
			return *result, nil
		}
	}
	return ScoreResult{}, fmt.Errorf("%w: score call returned %T", ErrInvalidOutput, result)
}

func noulResult(result Result) (NoulResult, error) {
	switch result := result.(type) {
	case NoulResult:
		return result, nil
	case *NoulResult:
		if result != nil {
			return *result, nil
		}
	}
	return NoulResult{}, fmt.Errorf("%w: noul call returned %T", ErrInvalidOutput, result)
}

type modelEngine interface {
	Predict(context.Context, State, []Question) (Prediction, error)
	Close(context.Context) error
}
