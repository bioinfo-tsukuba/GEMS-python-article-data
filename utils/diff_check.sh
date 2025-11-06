#!/usr/bin/env bash
# Do not abort mid-run: avoid -e / pipefail
set -u

groot="GEMS_paper_figures/data/241106_CCDS_253g1-hek293a_report/growth_curve"
proot="GEMS_paper_figures/data/241106_CCDS_253g1-hek293a_report/processed_data/curve_estimator_summary"

command -v jq >/dev/null || { echo "ERROR: jq is required (e.g. brew install jq)"; exit 2; }
command -v python3 >/dev/null || { echo "ERROR: python3 is required"; exit 2; }

# JSON normalisation: round numbers to 6 d.p., delete csv_path at all levels, and sort keys
norm_json() {
  local f="$1"
  jq -S '
    def walk(f):
      . as $in
      | if type == "object" then
          (reduce keys[] as $k ({}; . + { ($k): ($in[$k] | walk(f)) })) | f
        elif type == "array" then
          (map(walk(f))) | f
        else
          f
        end;
    walk(
      if type == "number" then ((. * 1000000) | round / 1000000)
      elif type == "object" then (del(.csv_path?) )
      else .
      end
    )
  ' "$f"
}

# CSV normalisation: round only numeric cells to 6 d.p. (leave NaN/Inf as strings)
norm_csv() {
  python3 - "$1" << 'PY'
import csv, sys, math
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

path = sys.argv[1]
def is_number(s: str) -> bool:
    s = s.strip()
    if not s: return False
    try:
        v = float(s)
        return not (math.isnan(v) or math.isinf(v))
    except ValueError:
        return False

def round6(s: str) -> str:
    q = Decimal('0.000001')
    d = Decimal(s)
    return str(d.quantize(q, rounding=ROUND_HALF_UP))

with open(path, newline='') as fp:
    reader = csv.reader(fp)
    writer = csv.writer(sys.stdout, lineterminator='\n')
    for row in reader:
        out = []
        for cell in row:
            c = cell.strip()
            if is_number(c):
                try:
                    out.append(round6(c))
                except (InvalidOperation, ValueError):
                    out.append(cell)
            else:
                out.append(cell)
        writer.writerow(out)
PY
}

status=0

echo "== JSON compare (normalised, csv_path ignored) =="
for pjson in "$proot"/*_shared_variable_history_fit.json; do
  base=$(basename "$pjson")
  gjson="$groot/${base%.json}_fit_params.json"
  if [[ ! -f "$gjson" ]]; then
    echo "MISSING(JSON): $gjson"
    status=1
    continue
  fi
  if ! diff -q <(norm_json "$pjson") <(norm_json "$gjson") >/dev/null; then
    echo "DIFF(JSON): $base"
    diff -u <(norm_json "$pjson") <(norm_json "$gjson") | sed -n '1,120p' || true
    status=1
  fi
done

echo
echo "== CSV compare (normalised) =="
for pcsv in "$proot"/*_shared_variable_history_fit.csv; do
  base=$(basename "$pcsv")
  gcsv="$groot/${base%.csv}_fit_params_fit.csv"
  if [[ ! -f "$gcsv" ]]; then
    echo "MISSING(CSV): $gcsv"
    status=1
    continue
  fi
  if ! diff -q <(norm_csv "$pcsv") <(norm_csv "$gcsv") >/dev/null; then
    echo "DIFF(CSV): $base"
    diff -u <(norm_csv "$pcsv") <(norm_csv "$gcsv") | sed -n '1,120p' || true
    status=1
  fi
done

if [[ $status -eq 0 ]]; then
  echo "OK: All files match (after normalisation; csv_path ignored)"
fi

exit $status
