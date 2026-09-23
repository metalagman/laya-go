package parity

import (
	_ "embed"
	"strings"
	"testing"
)

//go:embed testdata/corpus-v1.json
var corpusV1 []byte

const corpusV1SHA256 = "54a68e8a337fe05515e04a91d36d029c27564c425fd3821d4029787f9796065e"

func TestCorpusV1(t *testing.T) {
	t.Parallel()
	if err := VerifyDigest(corpusV1, corpusV1SHA256); err != nil {
		t.Fatal(err)
	}
	corpus, err := Load(corpusV1)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := len(corpus.Cases), 14; got != want {
		t.Fatalf("len(Cases) = %d, want %d", got, want)
	}
	if got, want := len(corpus.NegativeCases), 2; got != want {
		t.Fatalf("len(NegativeCases) = %d, want %d", got, want)
	}
	if got, want := len(corpus.Runs), 3; got != want {
		t.Fatalf("len(Runs) = %d, want %d", got, want)
	}
}

func TestCorpusV1RejectsIdentityToleranceAndContentChanges(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name string
		old  string
		new  string
	}{
		{"source", "052592a15d198d9ad47da779604259b10b47b7aa", "152592a15d198d9ad47da779604259b10b47b7aa"},
		{"profile", "2dc717bbbf2bacb9c00d57f2ba7279c6a4332900bca84d12c000ebd6401a0235", "3dc717bbbf2bacb9c00d57f2ba7279c6a4332900bca84d12c000ebd6401a0235"},
		{"exporter", "1087a4322ac2279653d5c132f0631246ab8770dd83f5be22ed7b21bd69a215c6", "2087a4322ac2279653d5c132f0631246ab8770dd83f5be22ed7b21bd69a215c6"},
		{"exporter-revision", "b61873c061ae62e52f4e0a6dc6fdafb6ffe88a88", "c61873c061ae62e52f4e0a6dc6fdafb6ffe88a88"},
		{"lock", "b08b67cdf27c9820ce9a4173a583b6da4706cfa72d9a60a333e58fc8a29b3e12", "a08b67cdf27c9820ce9a4173a583b6da4706cfa72d9a60a333e58fc8a29b3e12"},
		{"tolerance", `"atol":0.0001`, `"atol":0.0002`},
		{"content", `"id":"english-choice"`, `"id":"english-choicf"`},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			changed := strings.Replace(string(corpusV1), test.old, test.new, 1)
			if changed == string(corpusV1) {
				t.Fatal("test replacement did not change corpus")
			}
			if _, err := Load([]byte(changed)); err == nil {
				t.Fatal("Load(tampered corpus) succeeded")
			}
			if err := VerifyDigest([]byte(changed), corpusV1SHA256); err == nil {
				t.Fatal("VerifyDigest(tampered corpus) succeeded")
			}
		})
	}
}

func TestCorpusV1RejectsUnknownFields(t *testing.T) {
	t.Parallel()
	changed := strings.Replace(
		string(corpusV1),
		`"schema_version":1`,
		`"unknown":0,"schema_version":1`,
		1,
	)
	if _, err := Load([]byte(changed)); err == nil {
		t.Fatal("Load(corpus with unknown field) succeeded")
	}
}
