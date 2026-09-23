package laya_test

import (
	"os"
	"regexp"
	"strings"
	"testing"
)

func TestTaskfileDeclaresRequiredQualityGates(t *testing.T) {
	data, err := os.ReadFile("Taskfile.yml")
	if err != nil {
		t.Fatalf("os.ReadFile(Taskfile.yml) returned unexpected error: %v", err)
	}
	text := string(data)
	for _, task := range []string{
		"default", "help", "fmt", "tidy", "verify", "test", "test:race", "lint", "vuln", "check",
		"native:verify", "native:scan", "native:gate", "native:gate:race", "native:stress", "native:integration",
		"source:contract", "source:gate", "source:gate:race", "source:integration", "source:audit",
		"adk:contract", "adk:audit", "adk:gate", "adk:gate:race", "adk:integration",
		"example:contract", "example:audit", "example:gate", "example:gate:race", "example:integration",
		"qualification:gate",
		"release:sbom", "package:verify", "rollback:verify", "release:verify",
	} {
		if !strings.Contains(text, "  "+task+":\n") {
			t.Errorf("Taskfile.yml is missing task %q", task)
		}
	}
	for _, forbidden := range []string{"ignore_error: true", "download model", "huggingface", "rm -rf"} {
		if strings.Contains(strings.ToLower(text), forbidden) {
			t.Errorf("Taskfile.yml contains forbidden text %q", forbidden)
		}
	}
	for _, required := range []string{`GOPROXY: "off"`, `test ! -e ':memory:.ses'`} {
		if !strings.Contains(text, required) {
			t.Errorf("Taskfile.yml is missing native boundary check %q", required)
		}
	}
}

func TestCIUsesPinnedLeastPrivilegeQualityGate(t *testing.T) {
	data, err := os.ReadFile(".github/workflows/ci.yml")
	if err != nil {
		t.Fatalf("os.ReadFile(.github/workflows/ci.yml) returned unexpected error: %v", err)
	}
	text := string(data)

	for _, required := range []string{
		"permissions:\n  contents: read",
		"go-version: 1.26.6",
		"GOTOOLCHAIN: local",
		"run: task check",
	} {
		if !strings.Contains(text, required) {
			t.Errorf("ci.yml is missing required text %q", required)
		}
	}
	for _, forbidden := range []string{"continue-on-error", "permissions: write", "secrets.", "pull_request_target", "latest"} {
		if strings.Contains(text, forbidden) {
			t.Errorf("ci.yml contains forbidden text %q", forbidden)
		}
	}

	uses := regexp.MustCompile(`(?m)^\s*uses:\s*[^@\s]+@([^\s#]+)`).FindAllStringSubmatch(text, -1)
	if len(uses) == 0 {
		t.Fatal("ci.yml has no action references")
	}
	fullCommit := regexp.MustCompile(`^[0-9a-f]{40}$`)
	for _, use := range uses {
		if !fullCommit.MatchString(use[1]) {
			t.Errorf("ci.yml action reference %q is not pinned to a full commit SHA", use[0])
		}
	}
}
