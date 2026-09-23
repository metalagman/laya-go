// Package deps contains the reviewed native dependency identities.
package deps

import _ "embed"

// Manifest is the exact reviewed native dependency manifest.
//
//go:embed manifest.json
var Manifest []byte
