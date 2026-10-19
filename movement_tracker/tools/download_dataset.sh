#!/usr/bin/env bash
#
# Verifies, and where possible fetches, the CMU motion-capture dataset.
#
# =============================================================================
# READ THIS FIRST: the dataset is NOT fully re-downloadable automatically.
# =============================================================================
#
# CMU serves the original database as Asf/Amc/C3D/TVD files:
#
#     http://mocap.cs.cmu.edu/subjects/07/07.asf
#     http://mocap.cs.cmu.edu/subjects/07/07_01.amc
#
# This project reads BVH, not Asf/Amc. The BVH files used here are a converted
# distribution, and the converter that produced them (cgspeed.com) no longer
# resolves. There is therefore no automated path from the CMU originals to the exact
# files this project expects.
#
# What this script can honestly do:
#
#   --verify   check every file present against the recorded checksums, and report
#              exactly which ones are missing or altered
#   --fetch    fetch the CMU originals (Asf/Amc) so nothing is lost if the site
#              goes away. These are NOT usable by the test suite directly.
#
# For a new machine, copy training_data/ from a backup. Its contents are pinned in
# training_data/cmu_mocap.sha256, so a copy can be verified rather than trusted.
#
# A conversion tool could be written to rebuild BVH from Asf/Amc, but it is a
# substantial piece of work touching bone rest poses, joint order and unit scale.
# That is a project decision, not something to fake here.

set -euo pipefail

SUPPORT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$SUPPORT_DIR/training_data"
MANIFEST="$SUPPORT_DIR/tools/cmu_mocap.sha256"
CMU_BASE="http://mocap.cs.cmu.edu/subjects"

usage() {
  sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

if [[ ! -f "$MANIFEST" ]]; then
  echo "[FAIL] checksum manifest missing: $MANIFEST"
  echo "       Without it, a restored dataset cannot be verified."
  exit 2
fi

mode="${1:---verify}"
case "$mode" in
  --verify)
    total=0; ok=0; bad=0; missing=0
    bad_list=()
    while IFS= read -r line; do
      [[ -z "$line" ]] && continue
      expected="${line%% *}"
      rel="${line##* }"
      path="$DATA_DIR/../$rel"
      total=$((total + 1))
      if [[ ! -f "$path" ]]; then
        missing=$((missing + 1))
        continue
      fi
      actual=$(sha256sum "$path" | cut -d' ' -f1)
      if [[ "$actual" == "$expected" ]]; then
        ok=$((ok + 1))
      else
        bad=$((bad + 1))
        bad_list+=("$rel")
      fi
    done < "$MANIFEST"

    echo "dataset verification"
    echo "  expected files : $total"
    echo "  verified OK    : $ok"
    echo "  checksum mismatches: $bad"
    echo "  missing        : $missing"

    if (( bad > 0 )); then
      echo
      echo "  altered files:"
      for f in "${bad_list[@]}"; do echo "    - $f"; done
    fi

    if (( missing > 0 )); then
      echo
      echo "  $missing file(s) are absent. Copy them from a backup; a fresh clone"
      echo "  cannot regenerate them, because the BVH conversion source is gone."
    fi

    # Files come from a fixed, immutable archive. A single altered byte is a
    # finding, not a warning: every accuracy number in the project depends on these.
    if (( bad == 0 && missing == 0 )); then
      echo
      echo "[ OK ] dataset intact and verified against $total recorded checksums"
      exit 0
    fi
    exit 1
    ;;

  --fetch)
    # Archive the CMU originals. These are a safety net, not the operating data.
    out="$DATA_DIR/cmu_originals"
    mkdir -p "$out"
    echo "Fetching CMU originals into $out"
    echo "These are Asf/Amc and are NOT read by the test suite; they preserve the"
    echo "source material in case a BVH converter is written later."
    for subject in 007 008 009 013 014 020; do
      n=$((10#$subject))
      target="$out/$subject"
      mkdir -p "$target"
      code=$(curl -fsSL -o "$target/$n.asf" -w "%{http_code}" "$CMU_BASE/$subject/$n.asf" 2>/dev/null || echo "000")
      if [[ "$code" != "200" ]]; then
        echo "  [skip] subject $subject skeleton unavailable (HTTP $code)"
        rm -f "$target/$n.asf"
        continue
      fi
      echo "  [ ok ] $subject skeleton"
    done
    echo
    echo "Note: motion files (.amc) must be listed per subject from the search page."
    echo "This script fetches skeletons only; fetching every motion is left explicit"
    echo "so a partial run is obvious."
    ;;

  --help|-h|*)
    usage
    exit 0
    ;;
esac
