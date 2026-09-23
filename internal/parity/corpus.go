// Package parity verifies the frozen, synthetic upstream-parity corpus.
package parity

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"reflect"
)

const maxCorpusBytes = 2 << 20

// Corpus is the complete versioned parity fixture.
type Corpus struct {
	SchemaVersion int            `json:"schema_version"`
	Metadata      Metadata       `json:"metadata"`
	Inventory     Inventory      `json:"inventory"`
	Cases         []Case         `json:"cases"`
	NegativeCases []NegativeCase `json:"negative_cases"`
	Runs          []Run          `json:"runs"`
}

// Metadata binds the fixture to all accepted source and artifact identities.
type Metadata struct {
	SyntheticOnly        bool         `json:"synthetic_only"`
	SourceModel          Identity     `json:"source_model"`
	SDK                  SDK          `json:"sdk"`
	Profile              DigestID     `json:"profile"`
	Exporter             Exporter     `json:"exporter"`
	Lock                 FileDigest   `json:"lock"`
	PreprocessingVersion int          `json:"preprocessing_version"`
	BundleID             string       `json:"bundle_id"`
	ManifestSHA256       string       `json:"manifest_sha256"`
	BundleFiles          []BundleFile `json:"bundle_files"`
	Tolerance            Tolerance    `json:"tolerance"`
	Weights              Weights      `json:"weights"`
}

type Identity struct {
	ID       string `json:"id"`
	Revision string `json:"revision"`
}

type SDK struct {
	Repository string `json:"repository"`
	Revision   string `json:"revision"`
	Version    string `json:"version"`
}

type DigestID struct {
	ID     string `json:"id"`
	SHA256 string `json:"sha256"`
}

type Exporter struct {
	Name         string `json:"name"`
	Version      string `json:"version"`
	SourceSHA256 string `json:"source_sha256"`
	GitRevision  string `json:"git_revision"`
}

type FileDigest struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
}

type BundleFile struct {
	Role   string `json:"role"`
	Path   string `json:"path"`
	Size   int64  `json:"size"`
	SHA256 string `json:"sha256"`
}

type Tolerance struct {
	Absolute float64 `json:"atol"`
	Relative float64 `json:"rtol"`
}

type Weights struct {
	ParameterCount int64 `json:"parameter_count"`
	TensorCount    int   `json:"tensor_count"`
}

type Inventory struct {
	Categories      map[string][]string `json:"categories"`
	CaseIDs         []string            `json:"case_ids"`
	NegativeCaseIDs []string            `json:"negative_case_ids"`
	RunIDs          []string            `json:"run_ids"`
}

type Case struct {
	ID              string          `json:"id"`
	Categories      []string        `json:"categories"`
	State           State           `json:"state"`
	Question        Question        `json:"question"`
	RenderedOptions []string        `json:"rendered_options"`
	Run             RunLocation     `json:"run"`
	Sequence        Sequence        `json:"sequence"`
	Raw             OutputsRow      `json:"raw"`
	Calibration     Calibration     `json:"calibration"`
	Derived         Derived         `json:"derived"`
	TypedResult     json.RawMessage `json:"typed_result"`
	Usage           Usage           `json:"usage"`
}

type State struct {
	Kind  string `json:"kind"`
	Value string `json:"value"`
}

type Question struct {
	Type         string          `json:"t"`
	Instructions string          `json:"ins"`
	Criteria     json.RawMessage `json:"crit"`
}

type RunLocation struct {
	ID  string `json:"id"`
	Row int    `json:"row"`
}

type Sequence struct {
	InputIDs             []int64 `json:"input_ids"`
	MarkerPositions      []int64 `json:"marker_positions"`
	QType                int     `json:"qtype"`
	OriginalInputTokens  int     `json:"original_input_tokens"`
	EffectiveInputTokens int     `json:"effective_input_tokens"`
	Truncated            bool    `json:"truncated"`
}

type OutputsRow struct {
	Logits    []float64 `json:"logits"`
	ActLogits []float64 `json:"act_logits"`
}

type Calibration struct {
	RawTemperature       float64 `json:"raw_temperature"`
	EffectiveTemperature float64 `json:"effective_temperature"`
}

type Derived struct {
	Probabilities       []float64 `json:"probabilities"`
	ActionProbabilities []float64 `json:"action_probabilities"`
	EntropyConfidence   float64   `json:"entropy_confidence"`
	SelectedIndex       int       `json:"selected_index"`
	ExpectedScore       *float64  `json:"expected_score,omitempty"`
	TrueProbability     *float64  `json:"true_probability,omitempty"`
}

type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type NegativeCase struct {
	ID                  string   `json:"id"`
	Categories          []string `json:"categories"`
	SourceCaseID        string   `json:"source_case_id,omitempty"`
	QuestionOptionCount int      `json:"question_option_count,omitempty"`
	RetainedMarkerCount int      `json:"retained_marker_count,omitempty"`
	ExpectedError       string   `json:"expected_error"`
}

type Run struct {
	ID                        string             `json:"id"`
	CaseIDs                   []string           `json:"case_ids"`
	Tensors                   Tensors            `json:"tensors"`
	ReferenceOutputs          Outputs            `json:"reference_outputs"`
	ONNXOutputs               Outputs            `json:"onnx_outputs"`
	MaximumAbsoluteDifference map[string]float64 `json:"maximum_absolute_difference"`
	Usage                     Usage              `json:"usage"`
}

type Tensors struct {
	InputIDs      [][]int64 `json:"input_ids"`
	AttentionMask [][]int64 `json:"attention_mask"`
	MarkerPos     [][]int64 `json:"marker_pos"`
	MarkerMask    [][]bool  `json:"marker_mask"`
	QType         []int64   `json:"qtype"`
}

type Outputs struct {
	Logits    [][]float64 `json:"logits"`
	ActLogits [][]float64 `json:"act_logits"`
}

// Load strictly decodes and semantically verifies one corpus.
func Load(data []byte) (Corpus, error) {
	if len(data) == 0 || len(data) > maxCorpusBytes {
		return Corpus{}, fmt.Errorf("parity corpus size %d is outside bounds", len(data))
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var corpus Corpus
	if err := decoder.Decode(&corpus); err != nil {
		return Corpus{}, fmt.Errorf("decode parity corpus: %w", err)
	}
	if err := requireEOF(decoder); err != nil {
		return Corpus{}, err
	}
	if err := corpus.validate(); err != nil {
		return Corpus{}, err
	}
	return corpus, nil
}

// VerifyDigest checks the exact committed bytes before semantic parsing.
func VerifyDigest(data []byte, expected string) error {
	digest := sha256.Sum256(data)
	if hex.EncodeToString(digest[:]) != expected {
		return errors.New("parity corpus SHA-256 mismatch")
	}
	return nil
}

func requireEOF(decoder *json.Decoder) error {
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("parity corpus contains trailing JSON")
		}
		return fmt.Errorf("decode parity corpus trailer: %w", err)
	}
	return nil
}

func (corpus Corpus) validate() error {
	if corpus.SchemaVersion != 1 {
		return fmt.Errorf("unsupported parity corpus schema %d", corpus.SchemaVersion)
	}
	if err := corpus.Metadata.validate(); err != nil {
		return err
	}
	expectedCases := []string{
		"english-choice", "russian-score", "mixed-mask-noul", "unicode-choice",
		"two-options-choice", "six-options-choice", "eleven-options-choice",
		"json-null-noul", "json-bool-choice", "json-number-score", "json-string-noul",
		"long-options-choice", "exact-fit-choice", "right-truncated-choice",
	}
	expectedNegative := []string{"overflow-default-error", "impossible-marker-layout-error"}
	expectedRuns := []string{"ordered-mixed-batch", "coverage-short-batch", "boundary-batch"}
	if !reflect.DeepEqual(corpus.Inventory.CaseIDs, expectedCases) ||
		!reflect.DeepEqual(corpus.Inventory.NegativeCaseIDs, expectedNegative) ||
		!reflect.DeepEqual(corpus.Inventory.RunIDs, expectedRuns) {
		return errors.New("parity corpus inventory IDs do not match the reviewed set")
	}
	if len(corpus.Cases) != len(expectedCases) || len(corpus.NegativeCases) != len(expectedNegative) || len(corpus.Runs) != len(expectedRuns) {
		return errors.New("parity corpus inventory counts do not match")
	}
	cases := make(map[string]Case, len(corpus.Cases))
	categories := make(map[string][]string)
	for index, item := range corpus.Cases {
		if item.ID != expectedCases[index] {
			return fmt.Errorf("parity case %d has unexpected ID %q", index, item.ID)
		}
		if _, duplicate := cases[item.ID]; duplicate {
			return fmt.Errorf("duplicate parity case %q", item.ID)
		}
		if err := item.validate(); err != nil {
			return fmt.Errorf("parity case %q: %w", item.ID, err)
		}
		cases[item.ID] = item
		for _, category := range item.Categories {
			categories[category] = append(categories[category], item.ID)
		}
	}
	for index, item := range corpus.NegativeCases {
		if item.ID != expectedNegative[index] || item.ExpectedError != "input_too_long" {
			return fmt.Errorf("negative parity case %d does not match", index)
		}
		for _, category := range item.Categories {
			categories[category] = append(categories[category], item.ID)
		}
	}
	if corpus.NegativeCases[0].SourceCaseID != "right-truncated-choice" ||
		corpus.NegativeCases[1].QuestionOptionCount != 300 ||
		corpus.NegativeCases[1].RetainedMarkerCount >= 300 {
		return errors.New("negative parity boundary observations do not match")
	}
	if !reflect.DeepEqual(corpus.Inventory.Categories, categories) {
		return errors.New("parity category inventory does not match cases")
	}
	requiredCategories := []string{
		"english", "russian", "mixed-language", "unicode", "mask-replacement",
		"choice", "score", "noul", "option-count-2", "option-count-6",
		"option-count-11", "json-primitive", "batch", "boundary", "exact-fit",
		"overflow", "right-truncation", "long-options", "impossible-markers",
	}
	for _, category := range requiredCategories {
		if len(categories[category]) == 0 {
			return fmt.Errorf("required parity category %q is empty", category)
		}
	}
	for index, run := range corpus.Runs {
		if run.ID != expectedRuns[index] {
			return fmt.Errorf("parity run %d has unexpected ID %q", index, run.ID)
		}
		if err := run.validate(corpus.Metadata.Tolerance, cases); err != nil {
			return fmt.Errorf("parity run %q: %w", run.ID, err)
		}
	}
	return nil
}

func (metadata Metadata) validate() error {
	if !metadata.SyntheticOnly || metadata.PreprocessingVersion != 1 {
		return errors.New("parity metadata is not synthetic preprocessing v1")
	}
	if metadata.SourceModel != (Identity{ID: "convaiinnovations/laya-multilingual", Revision: "052592a15d198d9ad47da779604259b10b47b7aa"}) {
		return errors.New("parity source model identity mismatch")
	}
	if metadata.SDK != (SDK{Repository: "NandhaKishorM/laya", Revision: "573e5b62696ba441230cd6be71d593331b5d23af", Version: "0.3.5"}) {
		return errors.New("parity SDK identity mismatch")
	}
	if metadata.Profile != (DigestID{ID: "laya-multilingual-v1", SHA256: "2dc717bbbf2bacb9c00d57f2ba7279c6a4332900bca84d12c000ebd6401a0235"}) {
		return errors.New("parity profile identity mismatch")
	}
	if metadata.Exporter.Name != "laya-go-export" || metadata.Exporter.Version != "1.0.0" ||
		metadata.Exporter.SourceSHA256 != "c216a474624f970abbdf17673a94a6fb41cbbda036dbd8d9bcda91705fe1c136" {
		return errors.New("parity exporter identity mismatch")
	}
	if metadata.Lock != (FileDigest{Path: "tools/export/uv.lock", SHA256: "b08b67cdf27c9820ce9a4173a583b6da4706cfa72d9a60a333e58fc8a29b3e12"}) {
		return errors.New("parity lock identity mismatch")
	}
	const manifestDigest = "b3d35e00b0689988dff23f0eeed8721f10cce2c3a10d720f6c7d42cfc7d8c4de"
	if metadata.BundleID != "sha256:"+manifestDigest || metadata.ManifestSHA256 != manifestDigest {
		return errors.New("parity bundle identity mismatch")
	}
	if metadata.Tolerance != (Tolerance{Absolute: 1e-4, Relative: 1e-4}) {
		return errors.New("parity tolerance mismatch")
	}
	if metadata.Weights != (Weights{ParameterCount: 321908995, TensorCount: 170}) {
		return errors.New("parity weight observation mismatch")
	}
	if len(metadata.BundleFiles) != 9 {
		return errors.New("parity bundle file inventory mismatch")
	}
	seen := make(map[string]bool, len(metadata.BundleFiles))
	for _, file := range metadata.BundleFiles {
		if file.Path == "" || file.Role == "" || file.Size <= 0 || !validDigest(file.SHA256) || seen[file.Path] {
			return fmt.Errorf("invalid parity bundle file %q", file.Path)
		}
		seen[file.Path] = true
	}
	return nil
}

func (item Case) validate() error {
	if item.State.Kind != "text" && item.State.Kind != "json" {
		return errors.New("invalid state kind")
	}
	if item.State.Kind == "json" && !json.Valid([]byte(item.State.Value)) {
		return errors.New("invalid JSON primitive state")
	}
	qtypes := map[string]int{"choice": 0, "score": 1, "noul": 2}
	qtype, ok := qtypes[item.Question.Type]
	if !ok || item.Question.Instructions == "" || !json.Valid(item.Question.Criteria) {
		return errors.New("invalid question")
	}
	if item.Sequence.QType != qtype || item.Sequence.EffectiveInputTokens != len(item.Sequence.InputIDs) ||
		item.Sequence.EffectiveInputTokens > 1024 || item.Sequence.OriginalInputTokens < item.Sequence.EffectiveInputTokens {
		return errors.New("invalid sequence observation")
	}
	if item.Sequence.Truncated != (item.Sequence.OriginalInputTokens > item.Sequence.EffectiveInputTokens) {
		return errors.New("invalid truncation observation")
	}
	if len(item.RenderedOptions) < 2 || len(item.RenderedOptions) != len(item.Sequence.MarkerPositions) ||
		len(item.Raw.Logits) != len(item.RenderedOptions) || len(item.Raw.ActLogits) != 2 {
		return errors.New("invalid option or raw output shape")
	}
	if !finite(item.Raw.Logits...) || !finite(item.Raw.ActLogits...) ||
		!finite(item.Derived.Probabilities...) || !finite(item.Derived.ActionProbabilities...) {
		return errors.New("non-finite parity observation")
	}
	if len(item.Derived.Probabilities) != len(item.RenderedOptions) || len(item.Derived.ActionProbabilities) != 2 ||
		math.Abs(sum(item.Derived.Probabilities)-1) > 1e-12 || math.Abs(sum(item.Derived.ActionProbabilities)-1) > 1e-12 {
		return errors.New("invalid derived probability distribution")
	}
	if item.Derived.SelectedIndex < 0 || item.Derived.SelectedIndex >= len(item.RenderedOptions) ||
		item.Calibration.EffectiveTemperature < 0.5 || item.Calibration.EffectiveTemperature > 5 ||
		len(item.TypedResult) == 0 || item.Usage.InputTokens <= 0 || item.Usage.OutputTokens != 0 {
		return errors.New("invalid derived or typed observation")
	}
	return nil
}

func (run Run) validate(tolerance Tolerance, cases map[string]Case) error {
	rows := len(run.CaseIDs)
	if rows == 0 || len(run.Tensors.InputIDs) != rows || len(run.Tensors.AttentionMask) != rows ||
		len(run.Tensors.MarkerPos) != rows || len(run.Tensors.MarkerMask) != rows || len(run.Tensors.QType) != rows ||
		len(run.ReferenceOutputs.Logits) != rows || len(run.ReferenceOutputs.ActLogits) != rows ||
		len(run.ONNXOutputs.Logits) != rows || len(run.ONNXOutputs.ActLogits) != rows {
		return errors.New("batch row count mismatch")
	}
	inputTokens := 0
	for row, caseID := range run.CaseIDs {
		item, ok := cases[caseID]
		if !ok || item.Run != (RunLocation{ID: run.ID, Row: row}) {
			return fmt.Errorf("case location mismatch at row %d", row)
		}
		if len(run.Tensors.InputIDs[row]) != len(run.Tensors.AttentionMask[row]) ||
			len(run.Tensors.MarkerPos[row]) != len(run.Tensors.MarkerMask[row]) ||
			int(run.Tensors.QType[row]) != item.Sequence.QType {
			return fmt.Errorf("tensor shape mismatch at row %d", row)
		}
		for _, value := range run.Tensors.AttentionMask[row] {
			if value != 0 && value != 1 {
				return fmt.Errorf("invalid attention mask at row %d", row)
			}
			inputTokens += int(value)
		}
		if !allClose(run.ReferenceOutputs.Logits[row], run.ONNXOutputs.Logits[row], tolerance) ||
			!allClose(run.ReferenceOutputs.ActLogits[row], run.ONNXOutputs.ActLogits[row], tolerance) {
			return fmt.Errorf("ONNX parity mismatch at row %d", row)
		}
		optionCount := len(item.Sequence.MarkerPositions)
		if !reflect.DeepEqual(item.Raw.Logits, run.ReferenceOutputs.Logits[row][:optionCount]) ||
			!reflect.DeepEqual(item.Raw.ActLogits, run.ReferenceOutputs.ActLogits[row]) || item.Usage != run.Usage {
			return fmt.Errorf("case output mismatch at row %d", row)
		}
	}
	if run.Usage != (Usage{InputTokens: inputTokens, OutputTokens: 0}) {
		return errors.New("batch usage mismatch")
	}
	return nil
}

func allClose(left, right []float64, tolerance Tolerance) bool {
	if len(left) != len(right) || !finite(left...) || !finite(right...) {
		return false
	}
	for index := range left {
		if math.Abs(left[index]-right[index]) > tolerance.Absolute+tolerance.Relative*math.Abs(right[index]) {
			return false
		}
	}
	return true
}

func finite(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func sum(values []float64) float64 {
	var result float64
	for _, value := range values {
		result += value
	}
	return result
}

func validDigest(value string) bool {
	decoded, err := hex.DecodeString(value)
	return err == nil && len(decoded) == sha256.Size
}
