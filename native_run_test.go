package laya

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"sync"
	"testing"
)

func TestCancelableNativeCallCompletesAndCleans(t *testing.T) {
	call := newScriptedNativeCall()
	call.outputs = rawModelOutputs{logits: [][]float32{{1}}, actionLogits: [][]float32{{2, 3}}}
	close(call.allowCompletion)

	outputs, err := executeCancelableNativeCall(t.Context(), call)
	if err != nil {
		t.Fatalf("executeCancelableNativeCall returned unexpected error: %v", err)
	}
	if !reflect.DeepEqual(outputs, call.outputs) {
		t.Errorf("outputs = %+v, want %+v", outputs, call.outputs)
	}
	if got := call.eventSnapshot(); !reflect.DeepEqual(got, []string{"execute", "execute.done", "close"}) {
		t.Errorf("events = %v", got)
	}
	if call.terminateCalls != 0 || call.closeCalls != 1 {
		t.Errorf("calls = terminate %d, close %d", call.terminateCalls, call.closeCalls)
	}
}

func TestCancelableNativeCallTerminatesOnceAndJoins(t *testing.T) {
	runCause := errors.New("secret terminated run")
	terminateCause := errors.New("secret terminate detail")
	cleanupCause := errors.New("secret cleanup detail")
	call := newScriptedNativeCall()
	call.executeErr = runCause
	call.terminateErr = terminateCause
	call.closeErr = cleanupCause
	ctx, cancel := context.WithCancel(t.Context())
	done := make(chan error, 1)
	go func() {
		_, err := executeCancelableNativeCall(ctx, call)
		done <- err
	}()
	<-call.started
	cancel()
	err := <-done
	if !errors.Is(err, context.Canceled) {
		t.Errorf("error = %v, want errors.Is(_, context.Canceled)", err)
	}
	for _, cause := range []error{runCause, terminateCause, cleanupCause} {
		if !errors.Is(err, cause) {
			t.Errorf("error does not preserve cause %v", cause)
		}
	}
	if strings.Contains(err.Error(), "secret") {
		t.Errorf("error disclosed underlying detail: %v", err)
	}
	wantEvents := []string{"execute", "terminate", "execute.done", "close"}
	if got := call.eventSnapshot(); !reflect.DeepEqual(got, wantEvents) {
		t.Errorf("events = %v, want %v", got, wantEvents)
	}
	if call.terminateCalls != 1 || call.closeCalls != 1 {
		t.Errorf("calls = terminate %d, close %d", call.terminateCalls, call.closeCalls)
	}
}

func TestCancelableNativeCallCanceledBeforeExecution(t *testing.T) {
	call := newScriptedNativeCall()
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := executeCancelableNativeCall(ctx, call); !errors.Is(err, context.Canceled) {
		t.Errorf("error = %v, want errors.Is(_, context.Canceled)", err)
	}
	if got := call.eventSnapshot(); !reflect.DeepEqual(got, []string{"close"}) {
		t.Errorf("events = %v, want [close]", got)
	}
	if call.executeCalls != 0 || call.terminateCalls != 0 || call.closeCalls != 1 {
		t.Errorf(
			"calls = execute %d, terminate %d, close %d",
			call.executeCalls,
			call.terminateCalls,
			call.closeCalls,
		)
	}
}

func TestCancelableNativeCallNormalFailureCleansAfterJoin(t *testing.T) {
	runCause := errors.New("run failure")
	cleanupCause := errors.New("cleanup failure")
	call := newScriptedNativeCall()
	call.executeErr = runCause
	call.closeErr = cleanupCause
	close(call.allowCompletion)
	if _, err := executeCancelableNativeCall(t.Context(), call); !errors.Is(err, runCause) || !errors.Is(err, cleanupCause) {
		t.Errorf("error = %v, want both run and cleanup causes", err)
	}
	if got := call.eventSnapshot(); !reflect.DeepEqual(got, []string{"execute", "execute.done", "close"}) {
		t.Errorf("events = %v", got)
	}
}

type scriptedNativeCall struct {
	mu sync.Mutex

	started         chan struct{}
	allowCompletion chan struct{}
	events          []string
	outputs         rawModelOutputs
	executeErr      error
	terminateErr    error
	closeErr        error
	executeCalls    int
	terminateCalls  int
	closeCalls      int
}

func newScriptedNativeCall() *scriptedNativeCall {
	return &scriptedNativeCall{
		started:         make(chan struct{}),
		allowCompletion: make(chan struct{}),
	}
}

func (c *scriptedNativeCall) Execute() (rawModelOutputs, error) {
	c.mu.Lock()
	c.executeCalls++
	c.events = append(c.events, "execute")
	close(c.started)
	c.mu.Unlock()
	<-c.allowCompletion
	c.mu.Lock()
	c.events = append(c.events, "execute.done")
	c.mu.Unlock()
	return c.outputs, c.executeErr
}

func (c *scriptedNativeCall) Terminate() error {
	c.mu.Lock()
	c.terminateCalls++
	c.events = append(c.events, "terminate")
	c.mu.Unlock()
	select {
	case <-c.allowCompletion:
	default:
		close(c.allowCompletion)
	}
	return c.terminateErr
}

func (c *scriptedNativeCall) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.closeCalls++
	c.events = append(c.events, "close")
	return c.closeErr
}

func (c *scriptedNativeCall) eventSnapshot() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]string(nil), c.events...)
}
