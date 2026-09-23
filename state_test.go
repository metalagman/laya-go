package laya

import (
	"encoding/json"
	"errors"
	"testing"
)

func TestTextStatePreservesInput(t *testing.T) {
	input := "  exact state\n"
	state := TextState(input)

	if got, want := state.Kind(), StateKindText; got != want {
		t.Errorf("TextState(%q).Kind() = %v, want %v", input, got, want)
	}
	if got, ok := state.Text(); !ok || got != input {
		t.Errorf("TextState(%q).Text() = (%q, %t), want (%q, true)", input, got, ok, input)
	}
	if got, ok := state.JSON(); ok || got != nil {
		t.Errorf("TextState(%q).JSON() = (%q, %t), want (nil, false)", input, got, ok)
	}
	if err := state.validate(); err != nil {
		t.Errorf("TextState(%q).validate() = %v, want nil", input, err)
	}
}

func TestJSONStateCompactsAndCopies(t *testing.T) {
	input := []byte(" { \"b\" : 2, \"a\" : [1, true] } \n")
	state, err := JSONState(input)
	if err != nil {
		t.Fatalf("JSONState(%q) returned unexpected error: %v", input, err)
	}
	input[3] = 'x'

	if got, want := state.Kind(), StateKindJSON; got != want {
		t.Errorf("JSONState(input).Kind() = %v, want %v", got, want)
	}
	got, ok := state.JSON()
	if want := `{"b":2,"a":[1,true]}`; !ok || string(got) != want {
		t.Errorf("JSONState(input).JSON() = (%q, %t), want (%q, true)", got, ok, want)
	}

	got[2] = 'x'
	gotAgain, ok := state.JSON()
	if want := `{"b":2,"a":[1,true]}`; !ok || string(gotAgain) != want {
		t.Errorf("JSONState(input).JSON() after caller mutation = (%q, %t), want (%q, true)", gotAgain, ok, want)
	}
	if err := state.validate(); err != nil {
		t.Errorf("JSONState(input).validate() = %v, want nil", err)
	}
	if got, ok := state.Text(); ok || got != "" {
		t.Errorf("JSONState(input).Text() = (%q, %t), want (\"\", false)", got, ok)
	}
}

func TestJSONStateRejectsMalformedInput(t *testing.T) {
	inputs := [][]byte{
		nil,
		[]byte(`{"missing":`),
		[]byte(`{"one":1} {"two":2}`),
	}
	for _, input := range inputs {
		_, err := JSONState(input)
		if !errors.Is(err, ErrInvalidState) {
			t.Errorf("JSONState(%q) error = %v, want errors.Is(_, ErrInvalidState)", input, err)
		}
		var syntaxError *json.SyntaxError
		if !errors.As(err, &syntaxError) {
			t.Errorf("JSONState(%q) error = %v, want errors.As(_, *json.SyntaxError)", input, err)
		}
	}
}

func TestZeroStateIsInvalid(t *testing.T) {
	var state State
	if err := state.validate(); !errors.Is(err, ErrInvalidState) {
		t.Errorf("(State{}).validate() = %v, want errors.Is(_, ErrInvalidState)", err)
	}
}
