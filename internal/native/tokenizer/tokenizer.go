//go:build cgo && laya_native

// Package tokenizer provides the local-only subset of the pinned tokenizer
// binding used by Laya inference.
package tokenizer

/*
#cgo linux LDFLAGS: -ltokenizers -ldl -lm -lstdc++
#include <stdlib.h>
#include "tokenizers.h"

// Keep a relocation to the ABI marker exported by the pinned v1.27.0 static
// library. A mismatched older library therefore fails at link time.
void (*laya_tokenizers_version_check)(void) = &tokenizers_version_1_26_0;
*/
import "C"

import (
	"errors"
	"fmt"
	"math"
	"unsafe"
)

// ErrClosed indicates use of a nil or closed Tokenizer.
var ErrClosed = errors.New("tokenizer is closed")

// Tokenizer owns one native tokenizer handle. Calls must not overlap Close.
type Tokenizer struct {
	handle unsafe.Pointer
}

// FromBytes parses a complete local tokenizer.json document.
func FromBytes(data []byte) (*Tokenizer, error) {
	if len(data) == 0 {
		return nil, errors.New("tokenizer data is empty")
	}
	if len(data) > math.MaxUint32 {
		return nil, errors.New("tokenizer data exceeds native size limit")
	}
	options := C.struct_tokenizers_options{}
	var nativeErr *C.char
	handle := C.tokenizers_from_bytes(
		(*C.uint8_t)(unsafe.Pointer(&data[0])),
		C.uint32_t(len(data)),
		&options,
		&nativeErr,
	)
	return fromHandle(handle, nativeErr)
}

// FromFile parses a complete tokenizer.json at an already-local path.
func FromFile(path string) (*Tokenizer, error) {
	if path == "" {
		return nil, errors.New("tokenizer path is empty")
	}
	nativePath := C.CString(path)
	defer C.free(unsafe.Pointer(nativePath))

	var nativeErr *C.char
	handle := C.tokenizers_from_file(nativePath, &nativeErr)
	return fromHandle(handle, nativeErr)
}

func fromHandle(handle unsafe.Pointer, nativeErr *C.char) (*Tokenizer, error) {
	if nativeErr != nil {
		defer C.tokenizers_free_string(nativeErr)
		return nil, fmt.Errorf("parse tokenizer: %s", C.GoString(nativeErr))
	}
	if handle == nil {
		return nil, errors.New("parse tokenizer: native library returned nil")
	}
	return &Tokenizer{handle: handle}, nil
}

// Version returns the version reported by the linked native library.
func Version() string {
	return C.GoString(C.tokenizers_version())
}

// EncodeErr tokenizes text without any discovery or acquisition behavior.
func (t *Tokenizer) EncodeErr(text string, addSpecialTokens bool) ([]uint32, error) {
	if t == nil || t.handle == nil {
		return nil, ErrClosed
	}
	nativeText := C.CString(text)
	defer C.free(unsafe.Pointer(nativeText))

	options := C.struct_tokenizers_encode_options{
		add_special_tokens: C.bool(addSpecialTokens),
	}
	buffer := C.tokenizers_encode(t.handle, nativeText, &options)
	if buffer.len == 0 {
		if text != "" && buffer.ids == nil {
			return nil, errors.New("encode tokenizer input: native library returned no IDs")
		}
		return nil, nil
	}
	defer C.tokenizers_free_buffer(buffer)
	if buffer.ids == nil || uint64(buffer.len) > uint64(math.MaxInt) {
		return nil, errors.New("encode tokenizer input: invalid native buffer")
	}

	nativeIDs := unsafe.Slice((*C.uint32_t)(buffer.ids), int(buffer.len))
	ids := make([]uint32, len(nativeIDs))
	for index, id := range nativeIDs {
		ids[index] = uint32(id)
	}
	return ids, nil
}

// Close releases the native tokenizer. It is idempotent.
func (t *Tokenizer) Close() error {
	if t == nil || t.handle == nil {
		return nil
	}
	C.tokenizers_free_tokenizer(t.handle)
	t.handle = nil
	return nil
}
