//go:build laya_native && laya_native_diagnostics && cgo && linux

package laya

/*
#include <malloc.h>
*/
import "C"

// trimNativeAllocatorForTest releases free glibc arenas before protected RSS
// observations. It is unexported and omitted by dead-code elimination from
// binaries that do not use the protected diagnostics.
func trimNativeAllocatorForTest() {
	C.malloc_trim(0)
}
