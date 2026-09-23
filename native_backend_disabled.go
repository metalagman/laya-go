//go:build !laya_native || !cgo

package laya

import "context"

func newNativeRuntimeEngine(context.Context, RuntimeOptions) (runtimeEngine, error) {
	return nil, ErrNativeUnavailable
}
