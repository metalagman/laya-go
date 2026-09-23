// Package adklaya adapts an already-open Laya model to Google ADK workflows.
//
// The package performs local, in-process inference only. Callers own model
// acquisition and the lifetimes of the laya.Runtime and laya.Model supplied to
// constructors in this package.
package adklaya
