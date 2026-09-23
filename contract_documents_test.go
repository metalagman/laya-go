package laya_test

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"
)

const (
	officialModelID       = "convaiinnovations/laya-multilingual"
	officialModelRevision = "052592a15d198d9ad47da779604259b10b47b7aa"
	officialSDKRevision   = "573e5b62696ba441230cd6be71d593331b5d23af"
)

func TestBundleSchemaIsStrictJSON(t *testing.T) {
	schema := readJSONObject(t, "schema/bundle-manifest-v1.schema.json")
	if got := schema["$id"]; got != "urn:metalagman:laya-go:bundle-manifest:v1" {
		t.Errorf("schema $id = %v, want bundle manifest v1 URN", got)
	}
	assertObjectsRejectUnknownFields(t, "$", schema)

	properties := objectField(t, schema, "properties")
	for _, required := range []string{
		"schema_version", "bundle", "provenance", "preprocessing", "model",
		"tokenizer", "calibration", "runtime", "files",
	} {
		if _, ok := properties[required]; !ok {
			t.Errorf("schema properties missing %q", required)
		}
	}

	valid := readJSONObject(t, "schema/testdata/manifest-v1.valid.json")
	if got := numberField(t, valid, "schema_version"); got != 1 {
		t.Errorf("valid example schema_version = %v, want 1", got)
	}
	invalid := readJSONObject(t, "schema/testdata/manifest-v1.unknown-field.json")
	if _, ok := properties["unexpected"]; ok {
		t.Fatal("schema unexpectedly declares the negative example field")
	}
	if _, ok := invalid["unexpected"]; !ok {
		t.Fatal("negative example does not contain its intended unknown field")
	}
}

func TestOfficialExportProfileIdentity(t *testing.T) {
	profile := readJSONObject(t, "tools/export/profiles/laya-multilingual-v1.json")
	if got := profile["id"]; got != "laya-multilingual-v1" {
		t.Errorf("profile id = %v, want laya-multilingual-v1", got)
	}
	if got := numberField(t, profile, "profile_version"); got != 1 {
		t.Errorf("profile_version = %v, want 1", got)
	}

	source := objectField(t, profile, "source_model")
	if got := source["id"]; got != officialModelID {
		t.Errorf("source model id = %v, want %q", got, officialModelID)
	}
	if got := source["revision"]; got != officialModelRevision {
		t.Errorf("source model revision = %v, want %q", got, officialModelRevision)
	}
	sdk := objectField(t, profile, "reference_sdk")
	if got := sdk["revision"]; got != officialSDKRevision {
		t.Errorf("SDK revision = %v, want %q", got, officialSDKRevision)
	}
	licenses, ok := profile["bundle_licenses"].([]any)
	if !ok || len(licenses) != 2 {
		t.Fatalf("bundle_licenses = %T(%v), want two entries", profile["bundle_licenses"], profile["bundle_licenses"])
	}
	for i, value := range licenses {
		license, ok := value.(map[string]any)
		if !ok {
			t.Fatalf("bundle_licenses[%d] type = %T, want object", i, value)
		}
		for _, field := range []string{"component", "spdx", "license_path", "notice_path"} {
			if got, ok := license[field].(string); !ok || got == "" {
				t.Errorf("bundle_licenses[%d].%s = %v, want non-empty string", i, field, license[field])
			}
		}
	}

	files, ok := profile["source_files"].([]any)
	if !ok {
		t.Fatalf("source_files type = %T, want array", profile["source_files"])
	}
	if len(files) != 7 {
		t.Fatalf("source_files count = %d, want 7", len(files))
	}

	sha256Pattern := regexp.MustCompile(`^[0-9a-f]{64}$`)
	paths := make([]string, 0, len(files))
	seen := make(map[string]bool, len(files))
	for i, value := range files {
		file, ok := value.(map[string]any)
		if !ok {
			t.Fatalf("source_files[%d] type = %T, want object", i, value)
		}
		path, ok := file["path"].(string)
		if !ok || path == "" {
			t.Fatalf("source_files[%d].path = %v, want non-empty string", i, file["path"])
		}
		if seen[path] {
			t.Errorf("source_files repeats path %q", path)
		}
		seen[path] = true
		paths = append(paths, path)
		if size := numberField(t, file, "size"); size <= 0 {
			t.Errorf("source file %q size = %v, want positive", path, size)
		}
		digest, ok := file["sha256"].(string)
		if !ok || !sha256Pattern.MatchString(digest) {
			t.Errorf("source file %q sha256 = %v, want lowercase SHA-256", path, file["sha256"])
		}
	}
	if !slices.IsSorted(paths) {
		t.Errorf("source file paths are not sorted: %v", paths)
	}
}

func TestContractDocumentsExcludeCommunityInputs(t *testing.T) {
	paths := []string{
		"schema/bundle-manifest-v1.schema.json",
		"schema/testdata/manifest-v1.valid.json",
		"tools/export/profiles/laya-multilingual-v1.json",
		"docs/provenance/laya-multilingual-v1.json",
		"docs/bundle-format-v1.md",
		"docs/official-export-v1.md",
		"docs/upstream-parity-v1.md",
		"docs/source-operations.md",
		"docs/adk-operations.md",
		"docs/example-operations.md",
		"docs/qualification-v1.md",
		"docs/release-v0.1.md",
	}
	for _, path := range paths {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("os.ReadFile(%q): %v", path, err)
		}
		lower := strings.ToLower(string(data))
		for _, forbidden := range []string{"mizchi", "aac6fef", "community onnx", "community mlx"} {
			if strings.Contains(lower, forbidden) {
				t.Errorf("%s contains forbidden supported-input reference %q", path, forbidden)
			}
		}
	}
}

func TestSourceDocumentationMatchesImplementedContract(t *testing.T) {
	for _, path := range []string{"README.md", "doc.go", "docs/bundle-format-v1.md", "docs/source-operations.md"} {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("os.ReadFile(%q): %v", path, err)
		}
		text := string(data)
		for _, required := range []string{"complete", "WorkDir", "immutable"} {
			if !strings.Contains(text, required) {
				t.Errorf("%s does not describe %q", path, required)
			}
		}
		for _, stale := range []string{"returns `ErrUnsupportedSource`", "materialization is a later", "materialization and ADK integration remain later"} {
			if strings.Contains(text, stale) {
				t.Errorf("%s contains stale source claim %q", path, stale)
			}
		}
	}
}

func TestContractDocumentationUsesRawAndBundleTerms(t *testing.T) {
	for _, path := range []string{"docs/bundle-format-v1.md", "docs/official-export-v1.md"} {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("os.ReadFile(%q): %v", path, err)
		}
		text := string(data)
		for _, required := range []string{"raw", "bundle", "Safetensors"} {
			if !strings.Contains(text, required) {
				t.Errorf("%s does not define %q terminology", path, required)
			}
		}
	}
}

func readJSONObject(t *testing.T, path string) map[string]any {
	t.Helper()
	data, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		t.Fatalf("os.ReadFile(%q): %v", path, err)
	}
	var object map[string]any
	if err := json.Unmarshal(data, &object); err != nil {
		t.Fatalf("json.Unmarshal(%q): %v", path, err)
	}
	return object
}

func assertObjectsRejectUnknownFields(t *testing.T, path string, value any) {
	t.Helper()
	switch value := value.(type) {
	case map[string]any:
		if value["type"] == "object" && value["additionalProperties"] != false {
			t.Errorf("schema object %s does not set additionalProperties=false", path)
		}
		for key, child := range value {
			assertObjectsRejectUnknownFields(t, fmt.Sprintf("%s.%s", path, key), child)
		}
	case []any:
		for i, child := range value {
			assertObjectsRejectUnknownFields(t, fmt.Sprintf("%s[%d]", path, i), child)
		}
	}
}

func objectField(t *testing.T, object map[string]any, name string) map[string]any {
	t.Helper()
	value, ok := object[name].(map[string]any)
	if !ok {
		t.Fatalf("field %q type = %T, want object", name, object[name])
	}
	return value
}

func numberField(t *testing.T, object map[string]any, name string) float64 {
	t.Helper()
	value, ok := object[name].(float64)
	if !ok {
		t.Fatalf("field %q type = %T, want number", name, object[name])
	}
	return value
}
