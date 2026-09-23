package laya

import (
	"errors"
	"testing"
	"time"
)

func TestModelOptionsValidation(t *testing.T) {
	if err := (RuntimeOptions{}).validate(); err != nil {
		t.Errorf("(RuntimeOptions{}).validate() = %v, want nil", err)
	}
	tests := []struct {
		name    string
		options ModelOptions
		wantErr bool
	}{
		{name: "zero value"},
		{name: "bounded", options: ModelOptions{QueueCapacity: 4, MaxQueueWait: time.Second, InputTokenLimit: 512, Truncation: TruncateOverflow}},
		{name: "negative queue capacity", options: ModelOptions{QueueCapacity: -1}, wantErr: true},
		{name: "negative queue wait", options: ModelOptions{MaxQueueWait: -time.Second}, wantErr: true},
		{name: "negative input limit", options: ModelOptions{InputTokenLimit: -1}, wantErr: true},
		{name: "unknown truncation", options: ModelOptions{Truncation: TruncationPolicy(255)}, wantErr: true},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := test.options.validate()
			if gotErr := err != nil; gotErr != test.wantErr {
				t.Errorf("ModelOptions(%+v).validate() error = %v, want error presence = %t", test.options, err, test.wantErr)
			}
			if test.wantErr && !errors.Is(err, ErrInvalidConfig) {
				t.Errorf("ModelOptions(%+v).validate() error = %v, want errors.Is(_, ErrInvalidConfig)", test.options, err)
			}
		})
	}
}

func TestFSModelOptionsValidation(t *testing.T) {
	tests := []struct {
		name    string
		options FSModelOptions
		wantErr bool
	}{
		{name: "root default", options: FSModelOptions{WorkDir: "/tmp/laya"}},
		{name: "nested root", options: FSModelOptions{Root: "models/laya", WorkDir: "/tmp/laya"}},
		{name: "empty work directory", options: FSModelOptions{}, wantErr: true},
		{name: "escaping root", options: FSModelOptions{Root: "../laya", WorkDir: "/tmp/laya"}, wantErr: true},
		{name: "invalid model options", options: FSModelOptions{WorkDir: "/tmp/laya", ModelOptions: ModelOptions{QueueCapacity: -1}}, wantErr: true},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := test.options.validate()
			if gotErr := err != nil; gotErr != test.wantErr {
				t.Errorf("FSModelOptions(%+v).validate() error = %v, want error presence = %t", test.options, err, test.wantErr)
			}
			if test.wantErr && !errors.Is(err, ErrInvalidConfig) {
				t.Errorf("FSModelOptions(%+v).validate() error = %v, want errors.Is(_, ErrInvalidConfig)", test.options, err)
			}
		})
	}
}
