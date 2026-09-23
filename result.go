package laya

import (
	"fmt"
	"math"
)

// ModelID identifies the model reported by a prediction.
type ModelID string

// BundleID identifies the immutable model bundle reported by a prediction.
type BundleID string

// RuntimeID identifies the runtime instance reported by a prediction.
type RuntimeID string

// Usage contains token usage shared by all results in one Prediction.
type Usage struct {
	// InputTokens is the token count for the shared State and question batch.
	InputTokens int
	// OutputTokens is the token count for all results in the batch.
	OutputTokens int
}

// TruncationMetadata describes an explicitly enabled input truncation.
type TruncationMetadata struct {
	// Truncated reports whether the input was shortened under opt-in policy.
	Truncated bool
	// OriginalInputTokens is the input size observed before truncation.
	OriginalInputTokens int
	// EffectiveInputTokens is the input size evaluated after truncation.
	EffectiveInputTokens int
}

// PredictionMetadata identifies the execution and any visible truncation.
// Empty identities mean that the corresponding identity is unavailable; the
// library does not fabricate them.
type PredictionMetadata struct {
	// ModelID identifies the logical model when the engine can report it.
	ModelID ModelID
	// BundleID identifies the immutable model bundle when available.
	BundleID BundleID
	// RuntimeID identifies the native runtime instance when available.
	RuntimeID RuntimeID
	// Truncation describes any opt-in input shortening.
	Truncation TruncationMetadata
}

// Result is one of ChoiceResult, ScoreResult, or NoulResult.
//
// The interface is closed so a Prediction can exhaustively validate and retain
// the one-result-per-question contract.
type Result interface {
	QuestionID() QuestionID
	isResult()
}

// CriterionProbability associates one choice criterion with its probability.
type CriterionProbability struct {
	// CriterionID identifies the corresponding input criterion.
	CriterionID CriterionID
	// Probability is the calibrated probability in the inclusive range [0, 1].
	Probability float64
}

// ChoiceAnswer supplies the model values used to construct a ChoiceResult.
// Probabilities remain in criterion input order.
type ChoiceAnswer struct {
	// Selected identifies the chosen criterion.
	Selected CriterionID
	// Probabilities follow the input criterion order.
	Probabilities []CriterionProbability
	// Confidence is calibrated confidence in the primary choice.
	Confidence float64
	// ActionProbability is a distinct action or escalation probability.
	ActionProbability float64
}

// ChoiceResult contains a selected criterion and its ordered distribution.
// Confidence describes the primary choice; ActionProbability is a distinct
// action or escalation signal and must not be interpreted as confidence.
type ChoiceResult struct {
	questionID        QuestionID
	selected          CriterionID
	probabilities     []CriterionProbability
	confidence        float64
	actionProbability float64
}

// NewChoiceResult validates and copies a choice result.
func NewChoiceResult(questionID QuestionID, answer ChoiceAnswer) (ChoiceResult, error) {
	r := ChoiceResult{
		questionID:        questionID,
		selected:          answer.Selected,
		probabilities:     append([]CriterionProbability(nil), answer.Probabilities...),
		confidence:        answer.Confidence,
		actionProbability: answer.ActionProbability,
	}
	if err := r.validate(); err != nil {
		return ChoiceResult{}, err
	}
	return r, nil
}

// QuestionID returns the identifier of the answered question.
func (r ChoiceResult) QuestionID() QuestionID { return r.questionID }

// Selected returns the selected criterion identifier.
func (r ChoiceResult) Selected() CriterionID { return r.selected }

// Probabilities returns a copy of the criterion probabilities in input order.
func (r ChoiceResult) Probabilities() []CriterionProbability {
	return append([]CriterionProbability(nil), r.probabilities...)
}

// Confidence returns calibrated confidence in the primary choice.
func (r ChoiceResult) Confidence() float64 { return r.confidence }

// ActionProbability returns the distinct action or escalation probability.
func (r ChoiceResult) ActionProbability() float64 { return r.actionProbability }

func (ChoiceResult) isResult() {}

func (r ChoiceResult) validate() error {
	if r.questionID == "" {
		return fmt.Errorf("%w: choice result question ID is empty", ErrInvalidOutput)
	}
	if r.selected == "" {
		return fmt.Errorf("%w: choice result %q selected criterion is empty", ErrInvalidOutput, r.questionID)
	}
	if len(r.probabilities) == 0 {
		return fmt.Errorf("%w: choice result %q has no probabilities", ErrInvalidOutput, r.questionID)
	}
	seen := make(map[CriterionID]bool, len(r.probabilities))
	selectedFound := false
	for i, probability := range r.probabilities {
		if probability.CriterionID == "" {
			return fmt.Errorf("%w: choice result %q probability %d has an empty criterion ID", ErrInvalidOutput, r.questionID, i)
		}
		if seen[probability.CriterionID] {
			return fmt.Errorf("%w: choice result %q repeats criterion %q", ErrInvalidOutput, r.questionID, probability.CriterionID)
		}
		if err := validateProbability("choice probability", probability.Probability); err != nil {
			return fmt.Errorf("%w: choice result %q criterion %q: %w", ErrInvalidOutput, r.questionID, probability.CriterionID, err)
		}
		seen[probability.CriterionID] = true
		selectedFound = selectedFound || probability.CriterionID == r.selected
	}
	if !selectedFound {
		return fmt.Errorf("%w: choice result %q selected criterion %q is absent from probabilities", ErrInvalidOutput, r.questionID, r.selected)
	}
	if err := validateProbability("choice confidence", r.confidence); err != nil {
		return fmt.Errorf("%w: choice result %q: %w", ErrInvalidOutput, r.questionID, err)
	}
	if err := validateProbability("action probability", r.actionProbability); err != nil {
		return fmt.Errorf("%w: choice result %q: %w", ErrInvalidOutput, r.questionID, err)
	}
	return nil
}

// ScoreProbability associates an ordinal level and label with its probability.
type ScoreProbability struct {
	// Level is the zero-based ordinal position.
	Level int
	// Label matches the corresponding input rubric label.
	Label string
	// Probability is the calibrated probability in the inclusive range [0, 1].
	Probability float64
}

// ScoreAnswer supplies the model values used to construct a ScoreResult.
// Distribution remains in zero-based ordinal order.
type ScoreAnswer struct {
	// ExpectedLevel is the expected zero-based ordinal level.
	ExpectedLevel float64
	// Distribution follows the input rubric order.
	Distribution []ScoreProbability
	// Confidence is calibrated confidence in the primary score.
	Confidence float64
	// ActionProbability is a distinct action or escalation probability.
	ActionProbability float64
}

// ScoreResult contains an expected zero-based ordinal level and distribution.
// Confidence describes the primary score; ActionProbability is distinct.
type ScoreResult struct {
	questionID        QuestionID
	expectedLevel     float64
	distribution      []ScoreProbability
	confidence        float64
	actionProbability float64
}

// NewScoreResult validates and copies a score result.
func NewScoreResult(questionID QuestionID, answer ScoreAnswer) (ScoreResult, error) {
	r := ScoreResult{
		questionID:        questionID,
		expectedLevel:     answer.ExpectedLevel,
		distribution:      append([]ScoreProbability(nil), answer.Distribution...),
		confidence:        answer.Confidence,
		actionProbability: answer.ActionProbability,
	}
	if err := r.validate(); err != nil {
		return ScoreResult{}, err
	}
	return r, nil
}

// QuestionID returns the identifier of the answered question.
func (r ScoreResult) QuestionID() QuestionID { return r.questionID }

// ExpectedLevel returns the expected zero-based ordinal level.
func (r ScoreResult) ExpectedLevel() float64 { return r.expectedLevel }

// Distribution returns a copy of the labeled probabilities in ordinal order.
func (r ScoreResult) Distribution() []ScoreProbability {
	return append([]ScoreProbability(nil), r.distribution...)
}

// Confidence returns calibrated confidence in the primary score.
func (r ScoreResult) Confidence() float64 { return r.confidence }

// ActionProbability returns the distinct action or escalation probability.
func (r ScoreResult) ActionProbability() float64 { return r.actionProbability }

func (ScoreResult) isResult() {}

func (r ScoreResult) validate() error {
	if r.questionID == "" {
		return fmt.Errorf("%w: score result question ID is empty", ErrInvalidOutput)
	}
	if len(r.distribution) == 0 {
		return fmt.Errorf("%w: score result %q has no distribution", ErrInvalidOutput, r.questionID)
	}
	if math.IsNaN(r.expectedLevel) || math.IsInf(r.expectedLevel, 0) || r.expectedLevel < 0 || r.expectedLevel > float64(len(r.distribution)-1) {
		return fmt.Errorf("%w: score result %q expected level %v is outside [0,%d]", ErrInvalidOutput, r.questionID, r.expectedLevel, len(r.distribution)-1)
	}
	for i, probability := range r.distribution {
		if probability.Level != i {
			return fmt.Errorf("%w: score result %q distribution index %d has level %d", ErrInvalidOutput, r.questionID, i, probability.Level)
		}
		if probability.Label == "" {
			return fmt.Errorf("%w: score result %q level %d has an empty label", ErrInvalidOutput, r.questionID, i)
		}
		if err := validateProbability("score probability", probability.Probability); err != nil {
			return fmt.Errorf("%w: score result %q level %d: %w", ErrInvalidOutput, r.questionID, i, err)
		}
	}
	if err := validateProbability("score confidence", r.confidence); err != nil {
		return fmt.Errorf("%w: score result %q: %w", ErrInvalidOutput, r.questionID, err)
	}
	if err := validateProbability("action probability", r.actionProbability); err != nil {
		return fmt.Errorf("%w: score result %q: %w", ErrInvalidOutput, r.questionID, err)
	}
	return nil
}

// NoulResult contains calibrated P(true) for a boolean question.
// ActionProbability is a distinct action or escalation signal.
type NoulResult struct {
	questionID        QuestionID
	trueProbability   float64
	actionProbability float64
}

// NoulAnswer supplies the model values used to construct a NoulResult.
type NoulAnswer struct {
	// TrueProbability is calibrated P(true).
	TrueProbability float64
	// ActionProbability is a distinct action or escalation probability.
	ActionProbability float64
}

// NewNoulResult validates a boolean result.
func NewNoulResult(questionID QuestionID, answer NoulAnswer) (NoulResult, error) {
	r := NoulResult{
		questionID:        questionID,
		trueProbability:   answer.TrueProbability,
		actionProbability: answer.ActionProbability,
	}
	if err := r.validate(); err != nil {
		return NoulResult{}, err
	}
	return r, nil
}

// QuestionID returns the identifier of the answered question.
func (r NoulResult) QuestionID() QuestionID { return r.questionID }

// TrueProbability returns calibrated P(true).
func (r NoulResult) TrueProbability() float64 { return r.trueProbability }

// ActionProbability returns the distinct action or escalation probability.
func (r NoulResult) ActionProbability() float64 { return r.actionProbability }

func (NoulResult) isResult() {}

func (r NoulResult) validate() error {
	if r.questionID == "" {
		return fmt.Errorf("%w: noul result question ID is empty", ErrInvalidOutput)
	}
	if err := validateProbability("true probability", r.trueProbability); err != nil {
		return fmt.Errorf("%w: noul result %q: %w", ErrInvalidOutput, r.questionID, err)
	}
	if err := validateProbability("action probability", r.actionProbability); err != nil {
		return fmt.Errorf("%w: noul result %q: %w", ErrInvalidOutput, r.questionID, err)
	}
	return nil
}

// Prediction contains ordered results and usage shared by the whole request.
//
// Construct a Prediction with NewPrediction. Results returns a copy of the
// ordered result slice; concrete result values copy their own distributions.
type Prediction struct {
	results  []Result
	usage    Usage
	metadata PredictionMetadata
}

// NewPrediction validates and copies an ordered prediction.
func NewPrediction(results []Result, usage Usage, metadata PredictionMetadata) (Prediction, error) {
	p := Prediction{
		results:  append([]Result(nil), results...),
		usage:    usage,
		metadata: metadata,
	}
	if err := p.validate(); err != nil {
		return Prediction{}, err
	}
	return p, nil
}

// Results returns a copy of the results in question input order.
func (p Prediction) Results() []Result {
	return append([]Result(nil), p.results...)
}

// Usage returns token usage shared by the complete prediction.
func (p Prediction) Usage() Usage { return p.usage }

// Metadata returns model, bundle, runtime, and truncation observations.
func (p Prediction) Metadata() PredictionMetadata { return p.metadata }

func (p Prediction) validate() error {
	if len(p.results) == 0 {
		return fmt.Errorf("%w: prediction has no results", ErrInvalidOutput)
	}
	seen := make(map[QuestionID]bool, len(p.results))
	for i, result := range p.results {
		if err := validateResult(result); err != nil {
			return fmt.Errorf("prediction result %d: %w", i, err)
		}
		id := result.QuestionID()
		if seen[id] {
			return fmt.Errorf("%w: prediction repeats question ID %q", ErrInvalidOutput, id)
		}
		seen[id] = true
	}
	if p.usage.InputTokens < 0 || p.usage.OutputTokens < 0 {
		return fmt.Errorf("%w: prediction usage cannot be negative", ErrInvalidOutput)
	}
	return validateTruncation(p.metadata.Truncation)
}

func validateResult(result Result) error {
	switch result := result.(type) {
	case ChoiceResult:
		return result.validate()
	case *ChoiceResult:
		if result == nil {
			return fmt.Errorf("%w: choice result is nil", ErrInvalidOutput)
		}
		return result.validate()
	case ScoreResult:
		return result.validate()
	case *ScoreResult:
		if result == nil {
			return fmt.Errorf("%w: score result is nil", ErrInvalidOutput)
		}
		return result.validate()
	case NoulResult:
		return result.validate()
	case *NoulResult:
		if result == nil {
			return fmt.Errorf("%w: noul result is nil", ErrInvalidOutput)
		}
		return result.validate()
	default:
		return fmt.Errorf("%w: unsupported result type %T", ErrInvalidOutput, result)
	}
}

func validateProbability(name string, probability float64) error {
	if math.IsNaN(probability) || math.IsInf(probability, 0) || probability < 0 || probability > 1 {
		return fmt.Errorf("%s %v is outside [0,1]", name, probability)
	}
	return nil
}

func validateTruncation(metadata TruncationMetadata) error {
	if metadata.OriginalInputTokens < 0 || metadata.EffectiveInputTokens < 0 {
		return fmt.Errorf("%w: truncation token counts cannot be negative", ErrInvalidOutput)
	}
	if metadata.Truncated && metadata.OriginalInputTokens <= metadata.EffectiveInputTokens {
		return fmt.Errorf("%w: truncated input must reduce its token count", ErrInvalidOutput)
	}
	if !metadata.Truncated && metadata.OriginalInputTokens != metadata.EffectiveInputTokens {
		return fmt.Errorf("%w: untruncated input token counts differ", ErrInvalidOutput)
	}
	return nil
}
