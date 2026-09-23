package laya

import (
	"fmt"
	"math"
)

type rawModelOutputs struct {
	logits       [][]float32
	actionLogits [][]float32
}

type rowDerivation struct {
	probabilities       []float64
	actionProbabilities []float64
	confidence          float64
	expectedScore       float64
	trueProbability     float64
	temperature         calibrationValue
}

func formatInferencePrediction(
	batch preparedBatch,
	outputs rawModelOutputs,
	calibration calibrationConfig,
	metadata PredictionMetadata,
) (Prediction, error) {
	if err := validateRawOutputs(batch, outputs); err != nil {
		return Prediction{}, err
	}
	results := make([]Result, len(batch.items))
	for index, item := range batch.items {
		derivation, err := deriveOutputRow(
			item.question,
			outputs.logits[index][:len(item.question.options)],
			outputs.actionLogits[index],
			calibration,
		)
		if err != nil {
			return Prediction{}, fmt.Errorf("format output row %d: %w", index, err)
		}
		result, err := resultFromDerivation(item.question, derivation)
		if err != nil {
			return Prediction{}, fmt.Errorf("format output row %d: %w", index, err)
		}
		results[index] = result
	}
	metadata.Truncation = batch.truncation
	prediction, err := NewPrediction(results, batch.usage, metadata)
	if err != nil {
		return Prediction{}, fmt.Errorf("format prediction: %w", err)
	}
	return prediction, nil
}

func validateRawOutputs(batch preparedBatch, outputs rawModelOutputs) error {
	rows := len(batch.items)
	if len(outputs.logits) != rows || len(outputs.actionLogits) != rows {
		return fmt.Errorf("%w: output batch dimensions do not match %d input rows", ErrInvalidOutput, rows)
	}
	markerWidth := 0
	if rows > 0 {
		markerWidth = len(batch.markerPos[0])
	}
	for row := range rows {
		if len(outputs.logits[row]) != markerWidth {
			return fmt.Errorf("%w: logits row %d has width %d, want %d", ErrInvalidOutput, row, len(outputs.logits[row]), markerWidth)
		}
		if len(outputs.actionLogits[row]) != 2 {
			return fmt.Errorf("%w: action logits row %d has width %d, want 2", ErrInvalidOutput, row, len(outputs.actionLogits[row]))
		}
		for column, value := range outputs.logits[row] {
			if !isFinite(float64(value)) {
				return fmt.Errorf("%w: logits row %d column %d is not finite", ErrInvalidOutput, row, column)
			}
		}
		for column, value := range outputs.actionLogits[row] {
			if !isFinite(float64(value)) {
				return fmt.Errorf("%w: action logits row %d column %d is not finite", ErrInvalidOutput, row, column)
			}
		}
	}
	return nil
}

func deriveOutputRow(
	question preparedQuestion,
	logits []float32,
	actionLogits []float32,
	calibration calibrationConfig,
) (rowDerivation, error) {
	if len(logits) != len(question.options) {
		return rowDerivation{}, fmt.Errorf("%w: logit count does not match option count", ErrInvalidOutput)
	}
	temperature, err := calibration.selectValue(question)
	if err != nil {
		return rowDerivation{}, err
	}
	probabilities, err := stableSoftmax(logits, temperature.effective)
	if err != nil {
		return rowDerivation{}, err
	}
	actionProbabilities, err := stableSoftmax(actionLogits, 1)
	if err != nil {
		return rowDerivation{}, err
	}
	derivation := rowDerivation{
		probabilities:       probabilities,
		actionProbabilities: actionProbabilities,
		confidence:          confidenceFromProbabilities(probabilities),
		temperature:         temperature,
	}
	if question.kind == preparedScore {
		for index, probability := range probabilities {
			derivation.expectedScore += float64(index) * probability
		}
	}
	if question.kind == preparedNoul {
		derivation.trueProbability = probabilities[1]
	}
	return derivation, nil
}

func resultFromDerivation(question preparedQuestion, derivation rowDerivation) (Result, error) {
	actionProbability := roundFour(derivation.actionProbabilities[0])
	switch question.kind {
	case preparedChoice:
		best := maximumIndex(derivation.probabilities)
		probabilities := make([]CriterionProbability, len(question.optionIDs))
		for index, id := range question.optionIDs {
			probabilities[index] = CriterionProbability{
				CriterionID: CriterionID(id),
				Probability: roundFour(derivation.probabilities[index]),
			}
		}
		result, err := NewChoiceResult(question.id, ChoiceAnswer{
			Selected:          CriterionID(question.optionIDs[best]),
			Probabilities:     probabilities,
			Confidence:        roundFour(derivation.confidence),
			ActionProbability: actionProbability,
		})
		return result, err
	case preparedScore:
		distribution := make([]ScoreProbability, len(question.optionLabels))
		for index, label := range question.optionLabels {
			distribution[index] = ScoreProbability{
				Level:       index,
				Label:       label,
				Probability: roundFour(derivation.probabilities[index]),
			}
		}
		result, err := NewScoreResult(question.id, ScoreAnswer{
			ExpectedLevel:     roundFour(derivation.expectedScore),
			Distribution:      distribution,
			Confidence:        roundFour(derivation.confidence),
			ActionProbability: actionProbability,
		})
		return result, err
	case preparedNoul:
		result, err := NewNoulResult(question.id, NoulAnswer{
			TrueProbability:   roundFour(derivation.trueProbability),
			ActionProbability: actionProbability,
		})
		return result, err
	default:
		return nil, fmt.Errorf("%w: unsupported prepared question kind %d", ErrInvalidOutput, question.kind)
	}
}

func stableSoftmax[T ~float32 | ~float64](logits []T, temperature float64) ([]float64, error) {
	if len(logits) == 0 {
		return nil, fmt.Errorf("%w: logits are empty", ErrInvalidOutput)
	}
	if !isFinite(temperature) || temperature <= 0 {
		return nil, fmt.Errorf("%w: softmax temperature is invalid", ErrInvalidOutput)
	}
	scaled := make([]float64, len(logits))
	maximum := -math.MaxFloat64
	for index, logit := range logits {
		value := float64(logit)
		if !isFinite(value) {
			return nil, fmt.Errorf("%w: logit %d is not finite", ErrInvalidOutput, index)
		}
		scaled[index] = value / temperature
		maximum = max(maximum, scaled[index])
	}
	total := 0.0
	for index, value := range scaled {
		scaled[index] = math.Exp(value - maximum)
		total += scaled[index]
	}
	if !isFinite(total) || total <= 0 {
		return nil, fmt.Errorf("%w: softmax normalization is invalid", ErrInvalidOutput)
	}
	for index := range scaled {
		scaled[index] /= total
	}
	return scaled, nil
}

func confidenceFromProbabilities(probabilities []float64) float64 {
	if len(probabilities) < 2 {
		return 1
	}
	entropy := 0.0
	for _, probability := range probabilities {
		bounded := min(1.0, max(1e-12, probability))
		entropy -= probability * math.Log(bounded)
	}
	return min(1.0, max(0.0, 1-entropy/math.Log(float64(len(probabilities)))))
}

func maximumIndex(values []float64) int {
	best := 0
	for index := 1; index < len(values); index++ {
		if values[index] > values[best] {
			best = index
		}
	}
	return best
}

func roundFour(value float64) float64 {
	return math.Round(value*10_000) / 10_000
}
