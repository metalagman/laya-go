package laya

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"strconv"

	"github.com/metalagman/laya-go/internal/bundle"
)

type calibrationValue struct {
	raw       any
	effective float64
}

type calibrationConfig struct {
	byQType   []calibrationValue
	byOptions map[string]calibrationValue
}

const maxCalibrationConfigBytes = 1 << 20

type agentConfigDocument struct {
	Encoder              json.RawMessage            `json:"encoder"`
	HeadLayers           json.RawMessage            `json:"head_layers"`
	MaxLen               json.RawMessage            `json:"max_len"`
	HeadMaxLen           json.RawMessage            `json:"head_max_len"`
	MaxPrefixes          json.RawMessage            `json:"max_prefixes"`
	ActCosts             json.RawMessage            `json:"act_costs"`
	CostWrongAct         json.RawMessage            `json:"cost_wrong_act"`
	AMPDType             json.RawMessage            `json:"amp_dtype"`
	ModelName            json.RawMessage            `json:"model_name"`
	Temperature          []json.RawMessage          `json:"temperature"`
	TemperatureByOptions map[string]json.RawMessage `json:"temperature_by_options"`
	Training             json.RawMessage            `json:"training"`
}

func parseCalibrationConfig(manifest bundle.Manifest, data []byte) (calibrationConfig, error) {
	if len(data) == 0 || len(data) > maxCalibrationConfigBytes {
		return calibrationConfig{}, fmt.Errorf("%w: calibration config size is outside bounds", ErrInvalidBundle)
	}
	if err := rejectDuplicateCalibrationFields(data); err != nil {
		return calibrationConfig{}, newInferenceStageError(ErrInvalidBundle, "validate calibration config", err)
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var document agentConfigDocument
	if err := decoder.Decode(&document); err != nil {
		return calibrationConfig{}, newInferenceStageError(ErrInvalidBundle, "decode calibration config", err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return calibrationConfig{}, newInferenceStageError(ErrInvalidBundle, "decode calibration config trailer", err)
	}
	if len(document.Temperature) != 3 {
		return calibrationConfig{}, fmt.Errorf("%w: calibration temperature must contain three values", ErrUnsupportedBundle)
	}
	config := calibrationConfig{
		byQType:   make([]calibrationValue, len(document.Temperature)),
		byOptions: make(map[string]calibrationValue, len(document.TemperatureByOptions)),
	}
	for index, raw := range document.Temperature {
		config.byQType[index] = parseCalibrationValue(manifest, raw)
	}
	for bucket, raw := range document.TemperatureByOptions {
		config.byOptions[bucket] = parseCalibrationValue(manifest, raw)
	}
	return config, nil
}

func rejectDuplicateCalibrationFields(data []byte) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	if err := consumeCalibrationJSONValue(decoder, 0); err != nil {
		return err
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("additional JSON value")
		}
		return err
	}
	return nil
}

func consumeCalibrationJSONValue(decoder *json.Decoder, depth int) error {
	if depth > 64 {
		return errors.New("JSON nesting exceeds 64 levels")
	}
	token, err := decoder.Token()
	if err != nil {
		return err
	}
	delimiter, ok := token.(json.Delim)
	if !ok {
		return nil
	}
	switch delimiter {
	case '{':
		seen := make(map[string]bool)
		for decoder.More() {
			keyToken, err := decoder.Token()
			if err != nil {
				return err
			}
			key, ok := keyToken.(string)
			if !ok {
				return errors.New("JSON object key is not a string")
			}
			if seen[key] {
				return errors.New("duplicate JSON object field")
			}
			seen[key] = true
			if err := consumeCalibrationJSONValue(decoder, depth+1); err != nil {
				return err
			}
		}
		_, err = decoder.Token()
		return err
	case '[':
		for decoder.More() {
			if err := consumeCalibrationJSONValue(decoder, depth+1); err != nil {
				return err
			}
		}
		_, err = decoder.Token()
		return err
	default:
		return errors.New("invalid JSON delimiter")
	}
}

func parseCalibrationValue(manifest bundle.Manifest, raw json.RawMessage) calibrationValue {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		return calibrationValue{effective: manifest.Calibration.InvalidTemperature}
	}
	effective, ok := numericTemperature(value)
	if !ok || !isFinite(effective) {
		effective = manifest.Calibration.InvalidTemperature
	}
	effective = min(manifest.Calibration.TemperatureMax, max(manifest.Calibration.TemperatureMin, effective))
	return calibrationValue{raw: value, effective: effective}
}

func numericTemperature(value any) (float64, bool) {
	switch value := value.(type) {
	case json.Number:
		parsed, err := value.Float64()
		return parsed, err == nil
	case string:
		parsed, err := strconv.ParseFloat(value, 64)
		return parsed, err == nil
	case bool:
		if value {
			return 1, true
		}
		return 0, true
	case float64:
		return value, true
	case float32:
		return float64(value), true
	case int:
		return float64(value), true
	case int64:
		return float64(value), true
	default:
		return 0, false
	}
}

func (c calibrationConfig) selectValue(question preparedQuestion) (calibrationValue, error) {
	index := int(question.qtype)
	if index < 0 || index >= len(c.byQType) {
		return calibrationValue{}, fmt.Errorf("%w: calibration qtype %d is outside the configured range", ErrInvalidOutput, index)
	}
	bucket := temperatureBucket(questionKindName(question.kind), len(question.options))
	if value, ok := c.byOptions[bucket]; ok {
		return value, nil
	}
	return c.byQType[index], nil
}

func temperatureBucket(kind string, optionCount int) string {
	size := "11+"
	switch {
	case optionCount <= 2:
		size = "2"
	case optionCount <= 5:
		size = "3-5"
	case optionCount <= 10:
		size = "6-10"
	}
	return kind + ":" + size
}

func isFinite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}
