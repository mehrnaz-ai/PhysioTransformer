"""
data_quality_filter.py — Filter out bad samples before model training

Identifies and removes:
  1. Physiologically unrealistic values (HRmax > 240, LT < 70 bpm, etc.)
  2. Data entry errors (negative values, extreme outliers)
  3. Incomplete test data (too few stages, missing values)
  4. Suspicious lactate curves (peaks < 2 mmol/L, inverted trends)
  5. Athlete data inconsistencies (impossible age/BMI, weight extremes)

Output: Clean dataset with flagged issues documented
"""

import os
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ================================================================
# FILTERING THRESHOLDS (Based on Exercise Physiology)
# ================================================================

THRESHOLDS = {
    # Heart rate
    "hrmax_min": 100,           # Minimum realistic HRmax (bpm)
    "hrmax_max": 220,           # Maximum realistic HRmax (Karvonen: 220-age)
    "hr_min": 40,               # Minimum resting HR (bpm)
    "hr_per_stage_increase": 5, # Min HR increase per stage (should increase)

    # Lactate threshold
    "lt_min": 70,               # Min realistic LT (bpm) - very deconditioned
    "lt_max": 200,              # Max realistic LT (bpm)
    "lt_pct_hrmax_min": 0.40,   # LT should be at least 40% of HRmax
    "lt_pct_hrmax_max": 0.95,   # LT shouldn't exceed 95% of HRmax

    # Lactate values
    "lactate_min": 0.5,         # Minimum realistic lactate (mmol/L)
    "lactate_max": 20.0,        # Maximum realistic lactate (mmol/L)
    "lactate_peak_min": 2.0,    # Peak lactate should exceed 2 mmol/L
    "lactate_peak_max": 15.0,   # Peak lactate (realistic max)

    # Test stages
    "min_stages": 5,            # Minimum number of test stages
    "max_stages": 15,           # Maximum reasonable test stages

    # Power/metrics
    "power_negative": False,    # Power should never be negative
    "hr_negative": False,       # HR should never be negative

    # Athlete anthropometry
    "height_min": 140,          # cm (minimum realistic)
    "height_max": 220,          # cm (maximum realistic)
    "weight_min": 40,           # kg (minimum realistic)
    "weight_max": 150,          # kg (maximum realistic for endurance athlete)
    "bmi_min": 15,              # BMI threshold
    "bmi_max": 32,              # BMI threshold
    "age_min": 15,              # years (minimum realistic)
    "age_max": 80,              # years (maximum realistic)

    # Data consistency
    "max_consecutive_duplicates": 2,  # Max consecutive identical HR values
}

# ================================================================
# FILTER CLASS
# ================================================================

class DataQualityFilter:
    def __init__(self):
        self.issues = []
        self.removed_files = []
        self.kept_files = []
        self.stats = {}

    def check_file(self, file_path):
        """
        Check a single file for quality issues.
        Returns: (is_valid, issues_list)
        """
        issues = []

        try:
            sheets = pd.read_excel(file_path, sheet_name=None)
            if len(sheets) < 3:
                return False, ["Insufficient sheets (< 3)"]

            keys = list(sheets.keys())
            s1, s2, s3 = sheets[keys[0]], sheets[keys[1]], sheets[keys[2]]

            # ────────────────────────────────────────────────────────────
            # SHEET 1: METADATA
            # ────────────────────────────────────────────────────────────
            meta = {str(s1.iloc[i, 0]).lower(): s1.iloc[i, 1] for i in range(len(s1))}

            # Check height
            try:
                height = float(str(meta.get("height", "0")).replace(",", "."))
                if not (THRESHOLDS["height_min"] <= height <= THRESHOLDS["height_max"]):
                    issues.append(f"Height {height} cm out of range [{THRESHOLDS['height_min']}, {THRESHOLDS['height_max']}]")
            except:
                issues.append("Height: parsing error or missing")

            # Check weight
            try:
                weight = float(str(meta.get("weight", "0")).replace(",", "."))
                if not (THRESHOLDS["weight_min"] <= weight <= THRESHOLDS["weight_max"]):
                    issues.append(f"Weight {weight} kg out of range [{THRESHOLDS['weight_min']}, {THRESHOLDS['weight_max']}]")
            except:
                issues.append("Weight: parsing error or missing")

            # Check age (from DOB)
            try:
                dob = pd.to_datetime(meta.get("dob:", None), errors="coerce", dayfirst=True)
                test = pd.to_datetime(meta.get("test date:", None), errors="coerce", dayfirst=True)
                if pd.notna(dob) and pd.notna(test):
                    age = (test - dob).days / 365.25
                    if not (THRESHOLDS["age_min"] <= age <= THRESHOLDS["age_max"]):
                        issues.append(f"Age {age:.1f} years out of range [{THRESHOLDS['age_min']}, {THRESHOLDS['age_max']}]")
            except:
                issues.append("Age: parsing error or missing")

            # ────────────────────────────────────────────────────────────
            # SHEET 2: HR & LACTATE DATA
            # ────────────────────────────────────────────────────────────
            s2.columns = [str(c).lower() for c in s2.columns]

            if "hr" not in s2.columns or "bla" not in s2.columns:
                return False, ["Missing HR or Lactate columns"]

            hr = pd.to_numeric(s2["hr"], errors="coerce").values
            lac = pd.to_numeric(s2["bla"], errors="coerce").values

            # Check for valid data
            ok = ~np.isnan(hr) & ~np.isnan(lac)
            if ok.sum() < THRESHOLDS["min_stages"]:
                issues.append(f"Too few valid stages ({ok.sum()} < {THRESHOLDS['min_stages']})")
                return False, issues

            hr_valid = hr[ok]
            lac_valid = lac[ok]

            # Check number of stages
            n_stages = len(hr_valid)
            if not (THRESHOLDS["min_stages"] <= n_stages <= THRESHOLDS["max_stages"]):
                issues.append(f"Number of stages {n_stages} out of range [{THRESHOLDS['min_stages']}, {THRESHOLDS['max_stages']}]")

            # Check HR values
            hr_min = hr_valid.min()
            hr_max = hr_valid.max()

            if hr_min < THRESHOLDS["hr_min"]:
                issues.append(f"HR min {hr_min:.0f} below threshold {THRESHOLDS['hr_min']}")
            if hr_max > THRESHOLDS["hrmax_max"]:
                issues.append(f"HR max {hr_max:.0f} exceeds threshold {THRESHOLDS['hrmax_max']}")

            # Check HR increases monotonically
            hr_diff = np.diff(hr_valid)
            if np.any(hr_diff < THRESHOLDS["hr_per_stage_increase"]):
                issues.append(f"HR doesn't consistently increase ({np.sum(hr_diff < 5)} stages with <5 bpm increase)")

            # Check for consecutive duplicates
            hr_diff_zero = np.sum(hr_diff == 0)
            if hr_diff_zero > THRESHOLDS["max_consecutive_duplicates"]:
                issues.append(f"Excessive duplicate HR values ({hr_diff_zero} consecutive identical)")

            # Check lactate values
            if lac_valid.min() < THRESHOLDS["lactate_min"]:
                issues.append(f"Lactate min {lac_valid.min():.2f} below {THRESHOLDS['lactate_min']} mmol/L")
            if lac_valid.max() > THRESHOLDS["lactate_max"]:
                issues.append(f"Lactate max {lac_valid.max():.2f} exceeds {THRESHOLDS['lactate_max']} mmol/L")

            # Check peak lactate
            lac_peak = lac_valid.max()
            if lac_peak < THRESHOLDS["lactate_peak_min"]:
                issues.append(f"Peak lactate {lac_peak:.2f} < {THRESHOLDS['lactate_peak_min']} (test not maximal)")
            if lac_peak > THRESHOLDS["lactate_peak_max"]:
                issues.append(f"Peak lactate {lac_peak:.2f} > {THRESHOLDS['lactate_peak_max']} (suspiciously high)")

            # Check lactate trend (should generally increase)
            lac_diff = np.diff(lac_valid)
            if np.sum(lac_diff < -0.5) > len(lac_diff) * 0.3:  # More than 30% decreases
                issues.append(f"Lactate curve irregular (>30% decreases, poor quality)")

            # ────────────────────────────────────────────────────────────
            # SHEET 3: LT & EXTRA DATA
            # ────────────────────────────────────────────────────────────
            hrmax_ref = None
            hr_2mmol = None
            for i in range(len(s3)):
                k = str(s3.iloc[i, 0]).lower()
                try:
                    v = float(str(s3.iloc[i, 1]).replace(",", "."))
                    if "hrmax" in k:
                        hrmax_ref = v
                    if "hr @ 2mmol" in k:
                        hr_2mmol = v
                except:
                    pass

            # Use provided HRmax or measured max HR
            if hrmax_ref is not None and not np.isnan(hrmax_ref) and hrmax_ref >= 100:
                hrmax = float(hrmax_ref)
            else:
                hrmax = float(np.max(hr_valid))

            if not (THRESHOLDS["hrmax_min"] <= hrmax <= THRESHOLDS["hrmax_max"]):
                issues.append(f"HRmax {hrmax:.0f} out of realistic range [{THRESHOLDS['hrmax_min']}, {THRESHOLDS['hrmax_max']}]")

            # Check LT (lactate threshold)
            if hr_2mmol is not None and not np.isnan(hr_2mmol) and hr_2mmol > 0:
                lt_hr = float(hr_2mmol)
            else:
                # Interpolate at 2 mmol/L
                cross = np.where(lac_valid >= 2.0)[0]
                if len(cross) == 0:
                    lt_hr = hr_valid[-1]  # Use last HR if threshold not reached
                else:
                    i = cross[0]
                    if i == 0:
                        lt_hr = hr_valid[0]
                    else:
                        x1, x2 = hr_valid[i-1], hr_valid[i]
                        y1, y2 = lac_valid[i-1], lac_valid[i]
                        lt_hr = x1 + (2.0 - y1) * (x2 - x1) / (y2 - y1 + 1e-8)

            # Check LT in bpm
            if not (THRESHOLDS["lt_min"] <= lt_hr <= THRESHOLDS["lt_max"]):
                issues.append(f"LT {lt_hr:.0f} bpm out of range [{THRESHOLDS['lt_min']}, {THRESHOLDS['lt_max']}]")

            # Check LT as % of HRmax
            lt_pct = lt_hr / (hrmax + 1e-6)
            if not (THRESHOLDS["lt_pct_hrmax_min"] <= lt_pct <= THRESHOLDS["lt_pct_hrmax_max"]):
                issues.append(f"LT {lt_pct*100:.0f}% of HRmax out of range [{THRESHOLDS['lt_pct_hrmax_min']*100:.0f}, {THRESHOLDS['lt_pct_hrmax_max']*100:.0f}]%")

        except Exception as e:
            return False, [f"Exception during file processing: {str(e)[:100]}"]

        # ────────────────────────────────────────────────────────────
        # DECISION
        # ────────────────────────────────────────────────────────────
        if len(issues) == 0:
            return True, []
        else:
            # Some issues are warnings (file still usable)
            # Some issues are critical (file must be removed)

            # CRITICAL issues (file should be removed):
            critical_keywords = [
                "Insufficient sheets",
                "Missing HR or Lactate",
                "Too few valid stages",
                "parsing error or missing"
            ]

            has_critical = any(
                any(keyword in issue for keyword in critical_keywords)
                for issue in issues
            )

            if has_critical:
                return False, issues
            else:
                # Non-critical issues (file is usable but has warnings)
                return True, issues

    def filter_directory(self, root_dir):
        """
        Filter all files in directory.
        Returns: Summary of kept/removed files and issues.
        """
        files = [
            os.path.join(root_dir, f)
            for f in sorted(os.listdir(root_dir))
            if f.endswith(".xlsx")
        ]

        print(f"\n{'='*80}")
        print(f"DATA QUALITY FILTERING")
        print(f"{'='*80}")
        print(f"Total files to check: {len(files)}\n")

        quality_report = []

        for i, file_path in enumerate(files):
            fname = Path(file_path).stem
            is_valid, issues = self.check_file(file_path)

            status = "✓ KEEP" if is_valid else "✗ REMOVE"
            print(f"[{i+1:3d}/{len(files)}] {fname:<30} {status}")

            if issues:
                for issue in issues:
                    print(f"       → {issue}")
                    quality_report.append({
                        "file": fname,
                        "issue": issue,
                        "action": "KEEP (warning)" if is_valid else "REMOVE (critical)"
                    })

            if is_valid:
                self.kept_files.append(fname)
            else:
                self.removed_files.append(fname)

        return quality_report

    def print_summary(self):
        """Print filtering summary."""
        print(f"\n{'='*80}")
        print("FILTERING SUMMARY")
        print(f"{'='*80}")
        print(f"\nKept files:   {len(self.kept_files)}")
        print(f"Removed files: {len(self.removed_files)}")
        print(f"Total:        {len(self.kept_files) + len(self.removed_files)}")

        if self.removed_files:
            print(f"\n{'─'*80}")
            print("FILES TO REMOVE (Critical Issues):")
            print(f"{'─'*80}")
            for fname in self.removed_files:
                print(f"  ✗ {fname}.xlsx")

        print(f"\n{'─'*80}")
        print("KEPT FILES (Safe to Train):")
        print(f"{'─'*80}")
        for i, fname in enumerate(self.kept_files, 1):
            print(f"  {i:3d}. {fname}.xlsx")

        print(f"\n{'='*80}")
        print(f"RECOMMENDATION: Use {len(self.kept_files)} files ({len(self.kept_files)/(len(self.kept_files) + len(self.removed_files))*100:.1f}%) for training")
        print(f"Delete {len(self.removed_files)} files ({len(self.removed_files)/(len(self.kept_files) + len(self.removed_files))*100:.1f}%) with quality issues")
        print(f"{'='*80}\n")

    def save_report(self, output_file="data_quality_report.txt"):
        """Save detailed report to file."""
        with open(output_file, "w") as f:
            f.write("DATA QUALITY FILTERING REPORT\n")
            f.write("="*80 + "\n\n")

            f.write(f"FILTERING THRESHOLDS:\n")
            f.write("─"*80 + "\n")
            for key, value in THRESHOLDS.items():
                f.write(f"  {key:<30} = {value}\n")

            f.write(f"\n\nKEPT FILES ({len(self.kept_files)}):\n")
            f.write("─"*80 + "\n")
            for fname in sorted(self.kept_files):
                f.write(f"  {fname}.xlsx\n")

            f.write(f"\n\nREMOVED FILES ({len(self.removed_files)}):\n")
            f.write("─"*80 + "\n")
            for fname in sorted(self.removed_files):
                f.write(f"  {fname}.xlsx\n")

            f.write(f"\n\nDETAILED ISSUES:\n")
            f.write("─"*80 + "\n")
            for issue_dict in self.issues:
                f.write(f"{issue_dict}\n")

        print(f"Report saved to: {output_file}")


# ================================================================
# MAIN EXECUTION
# ================================================================

if __name__ == "__main__":
    import sys
    
    # Allow custom data directory
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"\n❌ ERROR: Directory '{data_dir}' not found")
        print(f"\nUsage:")
        print(f"  python data_quality_filter.py                    # uses ./data/")
        print(f"  python data_quality_filter.py /path/to/data      # custom path")
        print(f"\nPlease ensure your .xlsx files are in the correct directory.")
        sys.exit(1)
    
    # Create filter
    filter = DataQualityFilter()

    # Filter all files in data directory
    quality_report = filter.filter_directory(data_dir)

    # Print summary
    filter.print_summary()

    # Save detailed report
    filter.save_report("data_quality_report.txt")

    print("\n" + "="*80)
    print("ACTION ITEMS:")
    print("="*80)
    
    if filter.removed_files:
        print(f"\n1. DELETE {len(filter.removed_files)} files with critical quality issues:")
        print(f"   These files should be removed from your data/ directory:")
        for fname in sorted(filter.removed_files):
            print(f"     rm data/{fname}.xlsx")
    else:
        print(f"\n1. ✓ All {len(filter.kept_files)} files have acceptable quality!")
    
    print(f"\n2. RETRAIN MODELS with clean data:")
    if filter.removed_files:
        print(f"   (after deleting bad files, you'll have {len(filter.kept_files)} clean files)")
    print(f"   python model_full.py                  # Baseline")
    print(f"   python model_sport_attention.py       # Sport-specific attention (better option)")
    print(f"\n3. EVALUATE on clean data:")
    print(f"   python analysis_and_figures.py        # Generate figures")
    print(f"   python loso_validation.py             # Cross-sport transfer")
    print(f"   python baseline_comparison.py         # Compare to other models")
    print(f"\n" + "="*80)
