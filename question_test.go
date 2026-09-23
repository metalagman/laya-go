package laya

import (
	"errors"
	"testing"
)

func TestChoiceQuestionCopiesCriteria(t *testing.T) {
	criteria := []ChoiceCriterion{
		{ID: "safe", Description: "safe to proceed"},
		{ID: "review", Description: "requires review"},
	}
	question, err := NewChoiceQuestion("route", "Select the route.", criteria)
	if err != nil {
		t.Fatalf("NewChoiceQuestion() returned unexpected error: %v", err)
	}
	criteria[0].ID = "mutated"

	got := question.Criteria()
	if got[0].ID != "safe" {
		t.Errorf("question.Criteria()[0].ID = %q, want %q", got[0].ID, CriterionID("safe"))
	}
	got[0].ID = "also-mutated"
	if gotAgain := question.Criteria()[0].ID; gotAgain != "safe" {
		t.Errorf("question.Criteria()[0].ID after caller mutation = %q, want %q", gotAgain, CriterionID("safe"))
	}
	if got, want := question.ID(), QuestionID("route"); got != want {
		t.Errorf("question.ID() = %q, want %q", got, want)
	}
	if got, want := question.Instructions(), "Select the route."; got != want {
		t.Errorf("question.Instructions() = %q, want %q", got, want)
	}
}

func TestScoreQuestionCopiesRubric(t *testing.T) {
	rubric := []ScoreLevel{
		{Label: "low", Description: "low quality"},
		{Label: "high", Description: "high quality"},
	}
	question, err := NewScoreQuestion("quality", "Score quality.", rubric)
	if err != nil {
		t.Fatalf("NewScoreQuestion() returned unexpected error: %v", err)
	}
	rubric[0].Label = "mutated"

	got := question.Rubric()
	if got[0].Label != "low" {
		t.Errorf("question.Rubric()[0].Label = %q, want %q", got[0].Label, "low")
	}
	got[0].Label = "also-mutated"
	if gotAgain := question.Rubric()[0].Label; gotAgain != "low" {
		t.Errorf("question.Rubric()[0].Label after caller mutation = %q, want %q", gotAgain, "low")
	}
}

func TestNoulQuestionDescriptions(t *testing.T) {
	question, err := NewNoulQuestion("valid", "Is the document valid?", "invalid", "valid")
	if err != nil {
		t.Fatalf("NewNoulQuestion() returned unexpected error: %v", err)
	}
	if got, want := question.FalseDescription(), "invalid"; got != want {
		t.Errorf("question.FalseDescription() = %q, want %q", got, want)
	}
	if got, want := question.TrueDescription(), "valid"; got != want {
		t.Errorf("question.TrueDescription() = %q, want %q", got, want)
	}
}

func TestQuestionValidation(t *testing.T) {
	tests := []struct {
		name string
		call func() error
	}{
		{
			name: "empty ID",
			call: func() error {
				_, err := NewNoulQuestion("", "Valid?", "", "")
				return err
			},
		},
		{
			name: "empty instructions",
			call: func() error {
				_, err := NewNoulQuestion("valid", "", "", "")
				return err
			},
		},
		{
			name: "too few choice criteria",
			call: func() error {
				_, err := NewChoiceQuestion("route", "Route.", []ChoiceCriterion{{ID: "one", Description: "one"}})
				return err
			},
		},
		{
			name: "empty choice criterion ID",
			call: func() error {
				_, err := NewChoiceQuestion("route", "Route.", []ChoiceCriterion{{Description: "one"}, {ID: "two", Description: "two"}})
				return err
			},
		},
		{
			name: "empty choice description",
			call: func() error {
				_, err := NewChoiceQuestion("route", "Route.", []ChoiceCriterion{{ID: "one"}, {ID: "two", Description: "two"}})
				return err
			},
		},
		{
			name: "duplicate choice criterion",
			call: func() error {
				_, err := NewChoiceQuestion("route", "Route.", []ChoiceCriterion{{ID: "same", Description: "one"}, {ID: "same", Description: "two"}})
				return err
			},
		},
		{
			name: "too few score levels",
			call: func() error {
				_, err := NewScoreQuestion("score", "Score.", []ScoreLevel{{Label: "one", Description: "one"}})
				return err
			},
		},
		{
			name: "duplicate score label",
			call: func() error {
				_, err := NewScoreQuestion("score", "Score.", []ScoreLevel{{Label: "same", Description: "one"}, {Label: "same", Description: "two"}})
				return err
			},
		},
		{
			name: "empty score label",
			call: func() error {
				_, err := NewScoreQuestion("score", "Score.", []ScoreLevel{{Description: "one"}, {Label: "two", Description: "two"}})
				return err
			},
		},
		{
			name: "empty score description",
			call: func() error {
				_, err := NewScoreQuestion("score", "Score.", []ScoreLevel{{Label: "one"}, {Label: "two", Description: "two"}})
				return err
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if err := test.call(); !errors.Is(err, ErrInvalidRequest) {
				t.Errorf("constructor error = %v, want errors.Is(_, ErrInvalidRequest)", err)
			}
		})
	}
}

func TestValidateQuestionRejectsTypedNil(t *testing.T) {
	choice, err := NewChoiceQuestion("route", "Route.", []ChoiceCriterion{{ID: "one", Description: "one"}, {ID: "two", Description: "two"}})
	if err != nil {
		t.Fatalf("NewChoiceQuestion() returned unexpected error: %v", err)
	}
	score, err := NewScoreQuestion("score", "Score.", []ScoreLevel{{Label: "one", Description: "one"}, {Label: "two", Description: "two"}})
	if err != nil {
		t.Fatalf("NewScoreQuestion() returned unexpected error: %v", err)
	}
	noul, err := NewNoulQuestion("valid", "Valid?", "", "")
	if err != nil {
		t.Fatalf("NewNoulQuestion() returned unexpected error: %v", err)
	}

	valid := []Question{choice, &choice, score, &score, noul, &noul}
	for _, question := range valid {
		if err := validateQuestion(question); err != nil {
			t.Errorf("validateQuestion(%T) = %v, want nil", question, err)
		}
	}

	var nilChoice *ChoiceQuestion
	var nilScore *ScoreQuestion
	var nilNoul *NoulQuestion
	invalid := []Question{nilChoice, nilScore, nilNoul}
	for _, question := range invalid {
		if err := validateQuestion(question); !errors.Is(err, ErrInvalidRequest) {
			t.Errorf("validateQuestion(%T(nil)) = %v, want errors.Is(_, ErrInvalidRequest)", question, err)
		}
	}
}
