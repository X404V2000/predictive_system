#!/usr/bin/env bash
#
# Downloads the pose estimation models used by the webcam pipeline.
#
# The models are NOT committed to the repository. Two reasons:
#   * they are 17 MB of binaries, which do not belong in version control
#   * their provenance matters, and a checksum in a script documents where each
#     file came from and what it should contain far better than an opaque blob
#
# Both files come from the OpenCV Zoo project (Apache-2.0), which publishes them
# as ONNX specifically for use with OpenCV's DNN module.
#
# Usage:
#     bash tools/download_models.sh
#     bash tools/download_models.sh --check    # verify existing files only

set -euo pipefail

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models"
mkdir -p "$MODEL_DIR"

BASE="https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"

# name|url|sha256|expected_size_bytes
MODELS=(
  "pose_landmark.onnx|${BASE}/pose_estimation_mediapipe/pose_estimation_mediapipe_2023mar.onnx|9d89c599319a18fb7d2e28451a883476164543182bafca5f09eb2cf767ed2f3f|5557238"
  "person_detect.onnx|${BASE}/person_detection_mediapipe/person_detection_mediapipe_2023mar.onnx|47fd5599d6fa17608f03e0eb0ae230baa6e597d7e8a2c8199fe00abea55a701f|11990159"
)

CHECK_ONLY=0
if [[ "${1:-}" == "--check" ]]; then
  CHECK_ONLY=1
fi

verify() {
  local path="$1" expected_sha="$2" expected_size="$3"
  [[ -f "$path" ]] || return 1

  local actual_size actual_sha
  actual_size=$(stat -c%s "$path")
  if [[ "$actual_size" != "$expected_size" ]]; then
    echo "    size mismatch: got $actual_size, expected $expected_size"
    return 1
  fi

  actual_sha=$(sha256sum "$path" | cut -d' ' -f1)
  if [[ "$actual_sha" != "$expected_sha" ]]; then
    echo "    checksum mismatch"
    echo "      expected $expected_sha"
    echo "      got      $actual_sha"
    return 1
  fi
  return 0
}

failures=0
for entry in "${MODELS[@]}"; do
  IFS='|' read -r name url sha size <<< "$entry"
  dest="$MODEL_DIR/$name"
  human_size=$(( size / 1000000 ))

  if verify "$dest" "$sha" "$size"; then
    echo "[ OK ] $name already present and verified (${human_size} MB)"
    continue
  fi

  if [[ "$CHECK_ONLY" == "1" ]]; then
    echo "[MISS] $name is missing or failed verification"
    failures=$(( failures + 1 ))
    continue
  fi

  echo "[..] downloading $name (${human_size} MB)"
  tmp="$dest.partial"
  if ! curl -fsSL "$url" -o "$tmp"; then
    echo "[FAIL] could not download $name"
    rm -f "$tmp"
    failures=$(( failures + 1 ))
    continue
  fi

  if verify "$tmp" "$sha" "$size"; then
    mv "$tmp" "$dest"
    echo "[ OK ] $name verified"
  else
    echo "[FAIL] $name failed verification; discarding"
    rm -f "$tmp"
    failures=$(( failures + 1 ))
  fi
done

if [[ "$failures" -gt 0 ]]; then
  echo
  echo "$failures model(s) unavailable. The webcam source requires both."
  exit 1
fi

echo
echo "All models present and verified in $MODEL_DIR"
