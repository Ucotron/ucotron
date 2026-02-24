#!/usr/bin/env bash
# =============================================================================
# sync_versions.sh — Synchronize all package versions across SDKs
# =============================================================================
# Usage: ./scripts/sync_versions.sh <version>
# Example: ./scripts/sync_versions.sh 0.2.0
#
# Updates version in:
#   - Rust workspace (Cargo.toml [workspace.package] version)
#   - TypeScript SDK (package.json)
#   - Python SDK (pyproject.toml)
#   - Go SDK (no version file — uses git tags)
#   - Java SDK (build.gradle root + ucotron-sdk subproject)
#   - PHP SDK (composer.json)
# =============================================================================

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 0.2.0"
    exit 1
fi

VERSION="$1"
WORKSPACE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Validate semver format (X.Y.Z or X.Y.Z-prerelease)
if ! echo "$VERSION" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?$'; then
    echo "Error: Version '$VERSION' does not match semver format (X.Y.Z or X.Y.Z-pre)"
    exit 1
fi

echo "Synchronizing all packages to version $VERSION"
echo "================================================"

# Helper: cross-platform sed -i
sedi() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "$@"
    else
        sed -i "$@"
    fi
}

# --- Rust workspace version (all crates inherit via version.workspace = true) ---
WORKSPACE_TOML="$WORKSPACE_DIR/Cargo.toml"
if [ -f "$WORKSPACE_TOML" ]; then
    sedi "s/^version = \".*\"/version = \"$VERSION\"/" "$WORKSPACE_TOML"
    echo "  [Rust] Cargo.toml [workspace.package] → $VERSION"
fi

# --- TypeScript SDK ---
TS_PKG="$WORKSPACE_DIR/sdks/typescript/package.json"
if [ -f "$TS_PKG" ]; then
    sedi "s/\"version\": \".*\"/\"version\": \"$VERSION\"/" "$TS_PKG"
    echo "  [TypeScript] sdks/typescript/package.json → $VERSION"
fi

# --- Python SDK ---
PY_TOML="$WORKSPACE_DIR/sdks/python/pyproject.toml"
if [ -f "$PY_TOML" ]; then
    sedi "s/^version = \".*\"/version = \"$VERSION\"/" "$PY_TOML"
    echo "  [Python] sdks/python/pyproject.toml → $VERSION"
fi

# --- Go SDK (version is derived from git tag, no file to update) ---
echo "  [Go] Version derived from git tag v$VERSION (no file update needed)"

# --- Java SDK (root build.gradle + ucotron-sdk build.gradle) ---
JAVA_ROOT_GRADLE="$WORKSPACE_DIR/sdks/java/build.gradle"
if [ -f "$JAVA_ROOT_GRADLE" ]; then
    sedi "s/version = '.*'/version = '$VERSION'/" "$JAVA_ROOT_GRADLE"
    echo "  [Java] sdks/java/build.gradle → $VERSION"
fi

JAVA_SDK_GRADLE="$WORKSPACE_DIR/sdks/java/ucotron-sdk/build.gradle"
if [ -f "$JAVA_SDK_GRADLE" ]; then
    sedi "s/version = '.*'/version = '$VERSION'/" "$JAVA_SDK_GRADLE"
    echo "  [Java] sdks/java/ucotron-sdk/build.gradle → $VERSION"
fi

JAVA_ANDROID_GRADLE="$WORKSPACE_DIR/sdks/java/ucotron-sdk-android/build.gradle.kts"
if [ -f "$JAVA_ANDROID_GRADLE" ]; then
    sedi "s/version = \".*\"/version = \"$VERSION\"/" "$JAVA_ANDROID_GRADLE"
    echo "  [Java] sdks/java/ucotron-sdk-android/build.gradle.kts → $VERSION"
fi

# --- PHP SDK ---
PHP_COMPOSER="$WORKSPACE_DIR/sdks/php/composer.json"
if [ -f "$PHP_COMPOSER" ]; then
    if command -v jq >/dev/null 2>&1; then
        jq --indent 4 --arg v "$VERSION" '.version = $v' "$PHP_COMPOSER" > "$PHP_COMPOSER.tmp" && mv "$PHP_COMPOSER.tmp" "$PHP_COMPOSER"
    else
        sedi "s/\"version\": \".*\"/\"version\": \"$VERSION\"/" "$PHP_COMPOSER"
    fi
    echo "  [PHP] sdks/php/composer.json → $VERSION"
fi

echo ""
echo "Done! All packages synchronized to v$VERSION"
echo ""
echo "Next steps:"
echo "  1. Review changes: git diff"
echo "  2. Commit: git commit -am 'chore: bump version to $VERSION'"
echo "  3. Tag: git tag v$VERSION"
echo "  4. Push: git push && git push --tags"
