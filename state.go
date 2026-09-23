package laya

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// StateKind identifies the representation of a State.
type StateKind uint8

const (
	// StateKindInvalid identifies the zero State, which is not a valid request.
	StateKindInvalid StateKind = iota
	// StateKindText identifies state supplied as exact plain text.
	StateKindText
	// StateKindJSON identifies state supplied as a validated JSON value.
	StateKindJSON
)

// State is immutable input evaluated against one or more questions.
//
// Construct a State with TextState or JSONState. Its zero value is invalid.
type State struct {
	kind StateKind
	text string
	json []byte
}

// TextState returns a State that preserves text exactly, including whitespace.
func TextState(text string) State {
	return State{kind: StateKindText, text: text}
}

// JSONState validates data and returns a State containing its compact JSON
// representation. It copies data and does not retain caller-owned storage.
//
// JSONState returns an error matching ErrInvalidState for malformed JSON. The
// error also preserves the encoding/json parse error for errors.As.
func JSONState(data []byte) (State, error) {
	var compact bytes.Buffer
	if err := json.Compact(&compact, data); err != nil {
		return State{}, fmt.Errorf("%w: compact JSON state: %w", ErrInvalidState, err)
	}
	return State{kind: StateKindJSON, json: bytes.Clone(compact.Bytes())}, nil
}

// Kind returns the State representation.
func (s State) Kind() StateKind {
	return s.kind
}

// Text returns the exact text and reports whether s is text state.
func (s State) Text() (string, bool) {
	if s.kind != StateKindText {
		return "", false
	}
	return s.text, true
}

// JSON returns a copy of the compact JSON and reports whether s is JSON state.
func (s State) JSON() ([]byte, bool) {
	if s.kind != StateKindJSON {
		return nil, false
	}
	return bytes.Clone(s.json), true
}

func (s State) validate() error {
	switch s.kind {
	case StateKindInvalid:
		return fmt.Errorf("%w: use TextState or JSONState", ErrInvalidState)
	case StateKindText:
		return nil
	case StateKindJSON:
		if len(s.json) == 0 || !json.Valid(s.json) {
			return fmt.Errorf("%w: JSON state is empty or malformed", ErrInvalidState)
		}
		return nil
	default:
		return fmt.Errorf("%w: use TextState or JSONState", ErrInvalidState)
	}
}
