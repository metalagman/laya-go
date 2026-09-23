package laya

import "fmt"

// QuestionID identifies a question within a prediction request and result.
type QuestionID string

// CriterionID identifies a choice criterion within its question.
type CriterionID string

// Question is one of ChoiceQuestion, ScoreQuestion, or NoulQuestion.
//
// The interface is closed so the package can preserve exhaustive validation
// and ordered result matching as the API evolves.
type Question interface {
	ID() QuestionID
	Instructions() string
	isQuestion()
}

// ChoiceCriterion is one ordered candidate in a ChoiceQuestion.
type ChoiceCriterion struct {
	// ID identifies the criterion within its question.
	ID CriterionID
	// Description is the model-facing meaning of the criterion.
	Description string
}

// ChoiceQuestion asks the model to select one criterion.
//
// Construct a ChoiceQuestion with NewChoiceQuestion. The value is immutable;
// Criteria returns a copy of its ordered criteria.
type ChoiceQuestion struct {
	id           QuestionID
	instructions string
	criteria     []ChoiceCriterion
}

// NewChoiceQuestion validates and copies an ordered choice question.
func NewChoiceQuestion(id QuestionID, instructions string, criteria []ChoiceCriterion) (ChoiceQuestion, error) {
	q := ChoiceQuestion{
		id:           id,
		instructions: instructions,
		criteria:     append([]ChoiceCriterion(nil), criteria...),
	}
	if err := q.validate(); err != nil {
		return ChoiceQuestion{}, err
	}
	return q, nil
}

// ID returns the stable question identifier.
func (q ChoiceQuestion) ID() QuestionID { return q.id }

// Instructions returns the model-facing question instructions.
func (q ChoiceQuestion) Instructions() string { return q.instructions }

// Criteria returns a copy of the criteria in model input order.
func (q ChoiceQuestion) Criteria() []ChoiceCriterion {
	return append([]ChoiceCriterion(nil), q.criteria...)
}

func (ChoiceQuestion) isQuestion() {}

func (q ChoiceQuestion) validate() error {
	if err := validateQuestionHeader(q.id, q.instructions); err != nil {
		return err
	}
	if len(q.criteria) < 2 {
		return fmt.Errorf("%w: choice question %q has fewer than two criteria", ErrInvalidRequest, q.id)
	}
	seen := make(map[CriterionID]bool, len(q.criteria))
	for i, criterion := range q.criteria {
		if criterion.ID == "" {
			return fmt.Errorf("%w: choice question %q criterion %d has an empty ID", ErrInvalidRequest, q.id, i)
		}
		if criterion.Description == "" {
			return fmt.Errorf("%w: choice question %q criterion %q has an empty description", ErrInvalidRequest, q.id, criterion.ID)
		}
		if seen[criterion.ID] {
			return fmt.Errorf("%w: choice question %q repeats criterion %q", ErrInvalidRequest, q.id, criterion.ID)
		}
		seen[criterion.ID] = true
	}
	return nil
}

// ScoreLevel is one ordered, zero-based level in a ScoreQuestion rubric.
type ScoreLevel struct {
	// Label identifies the level in result distributions.
	Label string
	// Description is the model-facing meaning of the level.
	Description string
}

// ScoreQuestion asks the model for an ordinal score over an ordered rubric.
//
// Construct a ScoreQuestion with NewScoreQuestion. The value is immutable;
// Rubric returns a copy of its levels.
type ScoreQuestion struct {
	id           QuestionID
	instructions string
	rubric       []ScoreLevel
}

// NewScoreQuestion validates and copies an ordered score question.
func NewScoreQuestion(id QuestionID, instructions string, rubric []ScoreLevel) (ScoreQuestion, error) {
	q := ScoreQuestion{
		id:           id,
		instructions: instructions,
		rubric:       append([]ScoreLevel(nil), rubric...),
	}
	if err := q.validate(); err != nil {
		return ScoreQuestion{}, err
	}
	return q, nil
}

// ID returns the stable question identifier.
func (q ScoreQuestion) ID() QuestionID { return q.id }

// Instructions returns the model-facing question instructions.
func (q ScoreQuestion) Instructions() string { return q.instructions }

// Rubric returns a copy of the score levels in ordinal order.
func (q ScoreQuestion) Rubric() []ScoreLevel {
	return append([]ScoreLevel(nil), q.rubric...)
}

func (ScoreQuestion) isQuestion() {}

func (q ScoreQuestion) validate() error {
	if err := validateQuestionHeader(q.id, q.instructions); err != nil {
		return err
	}
	if len(q.rubric) < 2 {
		return fmt.Errorf("%w: score question %q has fewer than two levels", ErrInvalidRequest, q.id)
	}
	seen := make(map[string]bool, len(q.rubric))
	for i, level := range q.rubric {
		if level.Label == "" {
			return fmt.Errorf("%w: score question %q level %d has an empty label", ErrInvalidRequest, q.id, i)
		}
		if level.Description == "" {
			return fmt.Errorf("%w: score question %q level %d has an empty description", ErrInvalidRequest, q.id, i)
		}
		if seen[level.Label] {
			return fmt.Errorf("%w: score question %q repeats level label %q", ErrInvalidRequest, q.id, level.Label)
		}
		seen[level.Label] = true
	}
	return nil
}

// NoulQuestion asks the model for the probability that a statement is true.
// FalseDescription and TrueDescription may clarify the fixed false/true order.
type NoulQuestion struct {
	id               QuestionID
	instructions     string
	falseDescription string
	trueDescription  string
}

// NewNoulQuestion validates a boolean question. The descriptions are optional.
func NewNoulQuestion(id QuestionID, instructions, falseDescription, trueDescription string) (NoulQuestion, error) {
	q := NoulQuestion{
		id:               id,
		instructions:     instructions,
		falseDescription: falseDescription,
		trueDescription:  trueDescription,
	}
	if err := q.validate(); err != nil {
		return NoulQuestion{}, err
	}
	return q, nil
}

// ID returns the stable question identifier.
func (q NoulQuestion) ID() QuestionID { return q.id }

// Instructions returns the model-facing question instructions.
func (q NoulQuestion) Instructions() string { return q.instructions }

// FalseDescription returns the optional description of the false outcome.
func (q NoulQuestion) FalseDescription() string { return q.falseDescription }

// TrueDescription returns the optional description of the true outcome.
func (q NoulQuestion) TrueDescription() string { return q.trueDescription }

func (NoulQuestion) isQuestion() {}

func (q NoulQuestion) validate() error {
	return validateQuestionHeader(q.id, q.instructions)
}

func validateQuestionHeader(id QuestionID, instructions string) error {
	if id == "" {
		return fmt.Errorf("%w: question ID is empty", ErrInvalidRequest)
	}
	if instructions == "" {
		return fmt.Errorf("%w: question %q instructions are empty", ErrInvalidRequest, id)
	}
	return nil
}

func validateQuestion(q Question) error {
	switch q := q.(type) {
	case ChoiceQuestion:
		return q.validate()
	case *ChoiceQuestion:
		if q == nil {
			return fmt.Errorf("%w: choice question is nil", ErrInvalidRequest)
		}
		return q.validate()
	case ScoreQuestion:
		return q.validate()
	case *ScoreQuestion:
		if q == nil {
			return fmt.Errorf("%w: score question is nil", ErrInvalidRequest)
		}
		return q.validate()
	case NoulQuestion:
		return q.validate()
	case *NoulQuestion:
		if q == nil {
			return fmt.Errorf("%w: noul question is nil", ErrInvalidRequest)
		}
		return q.validate()
	default:
		return fmt.Errorf("%w: unsupported question type %T", ErrInvalidRequest, q)
	}
}
