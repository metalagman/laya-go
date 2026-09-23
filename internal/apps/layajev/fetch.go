package layajev

import (
	"context"
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
)

const (
	officialModelID  = "convaiinnovations/laya-multilingual"
	officialRevision = "052592a15d198d9ad47da779604259b10b47b7aa"
)

//go:embed pinned-source.json
var pinnedSourceProfile []byte

type sourceFile struct {
	Path   string `json:"path"`
	Size   int64  `json:"size"`
	SHA256 string `json:"sha256"`
}

type pinnedProfile struct {
	SourceModel struct {
		ID       string `json:"id"`
		Revision string `json:"revision"`
	} `json:"source_model"`
	SourceFiles []sourceFile `json:"source_files"`
}

func loadPinnedProfile(repositoryRoot string) (pinnedProfile, error) {
	data := pinnedSourceProfile
	if repositoryRoot != "" {
		path := filepath.Join(repositoryRoot, "tools", "export", "profiles", "laya-multilingual-v1.json")
		var err error
		data, err = os.ReadFile(path)
		if err != nil {
			return pinnedProfile{}, fmt.Errorf("fetch: read pinned profile: %w", err)
		}
	}
	var profile pinnedProfile
	if err := json.Unmarshal(data, &profile); err != nil {
		return pinnedProfile{}, fmt.Errorf("fetch: parse pinned profile: %w", err)
	}
	if profile.SourceModel.ID != officialModelID || profile.SourceModel.Revision != officialRevision || len(profile.SourceFiles) == 0 {
		return pinnedProfile{}, fmt.Errorf("fetch: profile is not the pinned official source")
	}
	seen := make(map[string]bool)
	for _, file := range profile.SourceFiles {
		if !validArtifactPath(file.Path) || seen[file.Path] || file.Size < 0 || file.Size > 4<<30 || len(file.SHA256) != 64 {
			return pinnedProfile{}, fmt.Errorf("fetch: invalid source file profile entry %q", file.Path)
		}
		if _, err := hex.DecodeString(file.SHA256); err != nil {
			return pinnedProfile{}, fmt.Errorf("fetch: invalid source digest for %q: %w", file.Path, err)
		}
		seen[file.Path] = true
	}
	return profile, nil
}

func fetchPinned(ctx context.Context, profile pinnedProfile, destination string, client *http.Client) error {
	return fetchPinnedFrom(ctx, profile, destination, client, "https://huggingface.co")
}

func fetchPinnedFrom(ctx context.Context, profile pinnedProfile, destination string, client *http.Client, baseURL string) (retErr error) {
	if destination == "" {
		return fmt.Errorf("fetch: destination is required")
	}
	if profile.SourceModel.ID != officialModelID || profile.SourceModel.Revision != officialRevision {
		return fmt.Errorf("fetch: source is not the pinned official model")
	}
	if _, err := os.Lstat(destination); err == nil {
		return fmt.Errorf("fetch: destination %q already exists", destination)
	} else if !os.IsNotExist(err) {
		return fmt.Errorf("fetch: inspect destination: %w", err)
	}
	parent := filepath.Dir(destination)
	if client == nil {
		client = http.DefaultClient
	}
	staging, err := os.MkdirTemp(parent, ".layajev-fetch-*")
	if err != nil {
		return fmt.Errorf("fetch: create staging directory: %w", err)
	}
	defer func() {
		if err := os.RemoveAll(staging); err != nil {
			retErr = errors.Join(retErr, fmt.Errorf("fetch: clean staging directory: %w", err))
		}
	}()
	for _, file := range profile.SourceFiles {
		if err := fetchFile(ctx, client, baseURL, staging, file); err != nil {
			return err
		}
	}
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("fetch: canceled before publication: %w", err)
	}
	if err := os.Rename(staging, destination); err != nil {
		return fmt.Errorf("fetch: publish verified snapshot: %w", err)
	}
	return nil
}

func fetchFile(ctx context.Context, client *http.Client, baseURL, staging string, file sourceFile) error {
	if !validArtifactPath(file.Path) || file.Size < 0 || file.Size > 4<<30 {
		return fmt.Errorf("fetch: invalid artifact path %q", file.Path)
	}
	url, err := url.Parse(strings.TrimRight(baseURL, "/"))
	if err != nil {
		return fmt.Errorf("fetch: base URL: %w", err)
	}
	url.Path = strings.TrimRight(url.Path, "/") + "/" + officialModelID + "/resolve/" + officialRevision + "/" + file.Path
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, url.String(), nil)
	if err != nil {
		return fmt.Errorf("fetch %q: %w", file.Path, err)
	}
	response, err := client.Do(request)
	if err != nil {
		return fmt.Errorf("fetch %q: %w", file.Path, err)
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("fetch %q: HTTP %d", file.Path, response.StatusCode)
	}
	path := filepath.Join(staging, filepath.FromSlash(file.Path))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("fetch %q: create artifact directory: %w", file.Path, err)
	}
	output, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("fetch %q: create artifact: %w", file.Path, err)
	}
	hash := sha256.New()
	count, copyErr := io.Copy(io.MultiWriter(output, hash), io.LimitReader(response.Body, file.Size+1))
	closeErr := output.Close()
	if copyErr != nil {
		return fmt.Errorf("fetch %q: copy artifact: %w", file.Path, copyErr)
	}
	if closeErr != nil {
		return fmt.Errorf("fetch %q: close artifact: %w", file.Path, closeErr)
	}
	if count != file.Size || hex.EncodeToString(hash.Sum(nil)) != file.SHA256 {
		return fmt.Errorf("fetch %q: size or SHA-256 mismatch", file.Path)
	}
	return nil
}

func validArtifactPath(path string) bool {
	return fs.ValidPath(path) && path != "." && !strings.Contains(path, `\`) && filepath.ToSlash(filepath.Clean(path)) == path
}
