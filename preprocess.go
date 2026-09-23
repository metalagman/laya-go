package laya

import (
	"errors"
	"fmt"
	"strings"

	"github.com/metalagman/laya-go/internal/bundle"
)

const maskTokenText = "<mask>"

type textTokenizer interface {
	Encode(string) ([]uint32, error)
}

type preparedQuestionKind uint8

const (
	preparedChoice preparedQuestionKind = iota + 1
	preparedScore
	preparedNoul
)

type preparedQuestion struct {
	id           QuestionID
	kind         preparedQuestionKind
	instructions string
	optionIDs    []string
	optionLabels []string
	options      []string
	qtype        int64
}

type preparedItem struct {
	question        preparedQuestion
	inputIDs        []int64
	markerPositions []int64
	originalTokens  int
	truncated       bool
}

type preparedBatch struct {
	inputIDs      [][]int64
	attentionMask [][]int64
	markerPos     [][]int64
	markerMask    [][]bool
	qtype         []int64
	items         []preparedItem
	usage         Usage
	truncation    TruncationMetadata
}

func prepareInferenceBatch(
	state State,
	questions []Question,
	manifest bundle.Manifest,
	tokenizer textTokenizer,
	opts ModelOptions,
) (preparedBatch, error) {
	if err := state.validate(); err != nil {
		return preparedBatch{}, err
	}
	if err := validateQuestions(questions); err != nil {
		return preparedBatch{}, err
	}
	if tokenizer == nil {
		return preparedBatch{}, fmt.Errorf("%w: tokenizer is nil", ErrNativeUnavailable)
	}
	if err := opts.validate(); err != nil {
		return preparedBatch{}, err
	}
	limit := manifest.Preprocessing.MaxLen
	if opts.InputTokenLimit > 0 && opts.InputTokenLimit < limit {
		limit = opts.InputTokenLimit
	}
	if limit <= 0 {
		return preparedBatch{}, fmt.Errorf("%w: input token limit is not positive", ErrInvalidConfig)
	}

	stateText, err := modelStateText(state)
	if err != nil {
		return preparedBatch{}, err
	}
	items := make([]preparedItem, len(questions))
	for index, question := range questions {
		prepared, err := normalizeQuestion(question, manifest)
		if err != nil {
			return preparedBatch{}, fmt.Errorf("prepare question %d: %w", index, err)
		}
		item, err := prepareItem(stateText, prepared, manifest, tokenizer, limit, opts.Truncation)
		if err != nil {
			return preparedBatch{}, fmt.Errorf("prepare question %d: %w", index, err)
		}
		items[index] = item
	}
	return collatePrepared(items, manifest.Tokenizer.SpecialTokens.Pad), nil
}

func modelStateText(state State) (string, error) {
	switch state.Kind() {
	case StateKindText:
		text, _ := state.Text()
		return text, nil
	case StateKindJSON:
		data, _ := state.JSON()
		return string(data), nil
	case StateKindInvalid:
		return "", fmt.Errorf("%w: unsupported state kind", ErrInvalidState)
	default:
		return "", fmt.Errorf("%w: unsupported state kind", ErrInvalidState)
	}
}

func normalizeQuestion(question Question, manifest bundle.Manifest) (preparedQuestion, error) {
	switch question := question.(type) {
	case ChoiceQuestion:
		return normalizeChoice(question, manifest), nil
	case *ChoiceQuestion:
		if question == nil {
			return preparedQuestion{}, fmt.Errorf("%w: choice question is nil", ErrInvalidRequest)
		}
		return normalizeChoice(*question, manifest), nil
	case ScoreQuestion:
		return normalizeScore(question, manifest), nil
	case *ScoreQuestion:
		if question == nil {
			return preparedQuestion{}, fmt.Errorf("%w: score question is nil", ErrInvalidRequest)
		}
		return normalizeScore(*question, manifest), nil
	case NoulQuestion:
		return normalizeNoul(question, manifest), nil
	case *NoulQuestion:
		if question == nil {
			return preparedQuestion{}, fmt.Errorf("%w: noul question is nil", ErrInvalidRequest)
		}
		return normalizeNoul(*question, manifest), nil
	default:
		return preparedQuestion{}, fmt.Errorf("%w: unsupported question type %T", ErrInvalidRequest, question)
	}
}

func normalizeChoice(question ChoiceQuestion, manifest bundle.Manifest) preparedQuestion {
	criteria := question.Criteria()
	optionIDs := make([]string, len(criteria))
	options := make([]string, len(criteria))
	for index, criterion := range criteria {
		optionIDs[index] = string(criterion.ID)
		if criterion.Description == "" {
			options[index] = string(criterion.ID)
		} else {
			options[index] = fmt.Sprintf("%s: %s", criterion.ID, criterion.Description)
		}
	}
	return preparedQuestion{
		id:           question.ID(),
		kind:         preparedChoice,
		instructions: question.Instructions(),
		optionIDs:    optionIDs,
		options:      options,
		qtype:        int64(manifest.Preprocessing.QTypes.Choice),
	}
}

func normalizeScore(question ScoreQuestion, manifest bundle.Manifest) preparedQuestion {
	rubric := question.Rubric()
	labels := make([]string, len(rubric))
	options := make([]string, len(rubric))
	for index, level := range rubric {
		labels[index] = level.Label
		options[index] = fmt.Sprintf("level %d: %s", index, level.Description)
	}
	return preparedQuestion{
		id:           question.ID(),
		kind:         preparedScore,
		instructions: question.Instructions(),
		optionLabels: labels,
		options:      options,
		qtype:        int64(manifest.Preprocessing.QTypes.Score),
	}
}

func normalizeNoul(question NoulQuestion, manifest bundle.Manifest) preparedQuestion {
	falseDescription := question.FalseDescription()
	if falseDescription == "" {
		falseDescription = "no, the statement does not hold"
	}
	trueDescription := question.TrueDescription()
	if trueDescription == "" {
		trueDescription = "yes, the statement holds"
	}
	return preparedQuestion{
		id:           question.ID(),
		kind:         preparedNoul,
		instructions: question.Instructions(),
		options: []string{
			"false: " + falseDescription,
			"true: " + trueDescription,
		},
		qtype: int64(manifest.Preprocessing.QTypes.Noul),
	}
}

func prepareItem(
	state string,
	question preparedQuestion,
	manifest bundle.Manifest,
	tokenizer textTokenizer,
	limit int,
	policy TruncationPolicy,
) (preparedItem, error) {
	preprocessing := manifest.Preprocessing
	kind := questionKindName(question.kind)
	instructions := strings.ReplaceAll(question.instructions, maskTokenText, " ")
	headIDs, err := encodeText(tokenizer, kind+" question: "+instructions, "instructions")
	if err != nil {
		return preparedItem{}, err
	}

	optionIDs := make([][]int64, len(question.options))
	for index, option := range question.options {
		encoded, err := encodeText(tokenizer, " "+strings.ReplaceAll(option, maskTokenText, " "), "option")
		if err != nil {
			return preparedItem{}, err
		}
		if len(encoded) > preprocessing.OptionMaxLen {
			encoded = encoded[:preprocessing.OptionMaxLen]
		}
		optionIDs[index] = append([]int64{int64(manifest.Tokenizer.SpecialTokens.Mask)}, encoded...)
	}

	optionBudget := preprocessing.HeadMaxLen - totalLength(optionIDs)
	if optionBudget < preprocessing.HeadReserve {
		perOption := max(4, (preprocessing.HeadMaxLen-preprocessing.HeadReserve)/max(1, len(optionIDs)))
		for index := range optionIDs {
			if len(optionIDs[index]) > perOption {
				optionIDs[index] = optionIDs[index][:perOption]
			}
		}
		optionBudget = preprocessing.HeadMaxLen - totalLength(optionIDs)
	}
	headLimit := max(8, optionBudget)
	if len(headIDs) > headLimit {
		headIDs = headIDs[:headLimit]
	}

	inputIDs := make([]int64, 0, 3+len(headIDs)+totalLength(optionIDs))
	inputIDs = append(inputIDs, int64(manifest.Tokenizer.SpecialTokens.CLS))
	inputIDs = append(inputIDs, headIDs...)
	inputIDs = append(inputIDs, int64(manifest.Tokenizer.SpecialTokens.SEP))
	markers := make([]int64, 0, len(optionIDs))
	for _, option := range optionIDs {
		markers = append(markers, int64(len(inputIDs)))
		inputIDs = append(inputIDs, option...)
	}
	inputIDs = append(inputIDs, int64(manifest.Tokenizer.SpecialTokens.SEP))

	stateIDs, err := encodeText(tokenizer, strings.ReplaceAll(state, maskTokenText, " "), "state")
	if err != nil {
		return preparedItem{}, err
	}
	originalTokens := len(inputIDs) + len(stateIDs) + 1
	if originalTokens > limit && policy == RejectOverflow {
		return preparedItem{}, &InputTooLongError{InputTokens: originalTokens, Limit: limit}
	}
	if len(inputIDs)+1 > limit {
		return preparedItem{}, &InputTooLongError{
			InputTokens: originalTokens,
			Limit:       limit,
			Err:         errors.New("question options do not fit within the input limit"),
		}
	}
	room := max(0, limit-len(inputIDs)-1)
	if len(stateIDs) > room {
		stateIDs = stateIDs[:room]
	}
	inputIDs = append(inputIDs, stateIDs...)
	inputIDs = append(inputIDs, int64(manifest.Tokenizer.SpecialTokens.SEP))
	if len(inputIDs) > limit {
		inputIDs = inputIDs[:limit]
	}
	retainedMarkers := markers[:0]
	for _, marker := range markers {
		if marker < int64(limit) {
			retainedMarkers = append(retainedMarkers, marker)
		}
	}
	if len(retainedMarkers) != len(question.options) {
		return preparedItem{}, &InputTooLongError{
			InputTokens: originalTokens,
			Limit:       limit,
			Err:         errors.New("option markers do not fit within the input limit"),
		}
	}

	return preparedItem{
		question:        question,
		inputIDs:        inputIDs,
		markerPositions: append([]int64(nil), retainedMarkers...),
		originalTokens:  originalTokens,
		truncated:       originalTokens > len(inputIDs),
	}, nil
}

func encodeText(tokenizer textTokenizer, text, stage string) ([]int64, error) {
	ids, err := tokenizer.Encode(text)
	if err != nil {
		return nil, newInferenceStageError(ErrNativeFailure, "tokenize "+stage, err)
	}
	encoded := make([]int64, len(ids))
	for index, id := range ids {
		encoded[index] = int64(id)
	}
	return encoded, nil
}

func questionKindName(kind preparedQuestionKind) string {
	switch kind {
	case preparedChoice:
		return "choice"
	case preparedScore:
		return "score"
	case preparedNoul:
		return "noul"
	default:
		return "invalid"
	}
}

func totalLength(values [][]int64) int {
	total := 0
	for _, value := range values {
		total += len(value)
	}
	return total
}

func collatePrepared(items []preparedItem, padID int) preparedBatch {
	sequenceLength := 0
	markerLength := 0
	for _, item := range items {
		sequenceLength = max(sequenceLength, len(item.inputIDs))
		markerLength = max(markerLength, len(item.markerPositions))
	}
	batch := preparedBatch{
		inputIDs:      make([][]int64, len(items)),
		attentionMask: make([][]int64, len(items)),
		markerPos:     make([][]int64, len(items)),
		markerMask:    make([][]bool, len(items)),
		qtype:         make([]int64, len(items)),
		items:         append([]preparedItem(nil), items...),
	}
	for index, item := range items {
		batch.inputIDs[index] = make([]int64, sequenceLength)
		for position := range batch.inputIDs[index] {
			batch.inputIDs[index][position] = int64(padID)
		}
		copy(batch.inputIDs[index], item.inputIDs)
		batch.attentionMask[index] = make([]int64, sequenceLength)
		for position := range item.inputIDs {
			batch.attentionMask[index][position] = 1
		}
		batch.markerPos[index] = make([]int64, markerLength)
		copy(batch.markerPos[index], item.markerPositions)
		batch.markerMask[index] = make([]bool, markerLength)
		for position := range item.markerPositions {
			batch.markerMask[index][position] = true
		}
		batch.qtype[index] = item.question.qtype
		batch.usage.InputTokens += len(item.inputIDs)
		batch.truncation.OriginalInputTokens += item.originalTokens
		batch.truncation.EffectiveInputTokens += len(item.inputIDs)
		batch.truncation.Truncated = batch.truncation.Truncated || item.truncated
	}
	return batch
}

type inferenceStageError struct {
	category error
	stage    string
	cause    error
}

func newInferenceStageError(category error, stage string, cause error) error {
	return &inferenceStageError{category: category, stage: stage, cause: cause}
}

func (e *inferenceStageError) Error() string {
	if e == nil {
		return "laya: inference failure"
	}
	return e.category.Error() + ": " + e.stage
}

func (e *inferenceStageError) Is(target error) bool {
	return e != nil && target == e.category
}

func (e *inferenceStageError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.cause
}
