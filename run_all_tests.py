#!/usr/bin/env python3
"""
Run all voice conversion tests and generate comparison report
"""

import subprocess
import sys
import os
import json
from datetime import datetime


def run_test(script_name, description):
    """Run a test script and capture output"""

    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*70}\n")

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        print(result.stdout)

        if result.returncode != 0:
            print(f"ERROR in {script_name}:")
            print(result.stderr)
            return False

        return True

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {script_name} took too long")
        return False
    except Exception as e:
        print(f"EXCEPTION running {script_name}: {e}")
        return False


def generate_report():
    """Generate markdown report from results"""

    report = []
    report.append("# Voice Conversion Test Results\n")
    report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append("---\n\n")

    report.append("## Test Configuration\n\n")
    report.append("- **Platform**: CPU-only\n")
    report.append("- **Target**: ≤2MB memory, <100ms latency, RTF <1.0\n")
    report.append("- **Test Audio**: Synthetic male (120Hz) and female (220Hz) voices\n")
    report.append("- **Duration**: 3 seconds per sample\n\n")
    report.append("---\n\n")

    # Check which results exist
    methods = {
        'WORLD Vocoder': 'results/world',
        'PSOLA': 'results/psola'
    }

    report.append("## Results Summary\n\n")
    report.append("| Method | Memory (MB) | Latency (ms) | RTF | Status |\n")
    report.append("|--------|-------------|--------------|-----|--------|\n")

    for method_name, result_dir in methods.items():
        if os.path.exists(result_dir):
            # Count output files
            files = os.listdir(result_dir)
            num_files = len([f for f in files if f.endswith('.wav')])
            status = f"✓ {num_files} files" if num_files > 0 else "✗ No files"

            report.append(f"| {method_name} | See detailed results | See detailed results | See detailed results | {status} |\n")
        else:
            report.append(f"| {method_name} | - | - | - | ✗ Not run |\n")

    report.append("\n---\n\n")

    report.append("## Detailed Results\n\n")
    report.append("See test output above for detailed metrics including:\n\n")
    report.append("- Memory profiling (Python peak and system delta)\n")
    report.append("- Latency measurements (average ± std dev)\n")
    report.append("- Real-time factor (RTF)\n")
    report.append("- Pitch shift accuracy\n\n")

    report.append("## Generated Files\n\n")

    for method_name, result_dir in methods.items():
        if os.path.exists(result_dir):
            report.append(f"### {method_name}\n\n")
            files = sorted([f for f in os.listdir(result_dir) if f.endswith('.wav')])
            if files:
                for f in files:
                    report.append(f"- `{result_dir}/{f}`\n")
            else:
                report.append("No output files generated\n")
            report.append("\n")

    report.append("---\n\n")
    report.append("## Next Steps\n\n")
    report.append("1. Listen to output files in `results/` directories\n")
    report.append("2. Compare quality subjectively\n")
    report.append("3. Verify memory and latency meet constraints\n")
    report.append("4. Select best approach for your use case\n\n")

    report.append("**Note**: For ML/DL approaches (TinyVC), model training and quantization are required before testing.\n")

    return ''.join(report)


def main():
    """Run all tests"""

    print("=" * 70)
    print("Voice Conversion - Comprehensive Test Suite")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Create results directory
    os.makedirs('results', exist_ok=True)

    tests = [
        ('test_world_vocoder.py', 'WORLD Vocoder Test'),
        ('test_psola.py', 'PSOLA Test')
    ]

    results = {}

    for script, description in tests:
        if os.path.exists(script):
            success = run_test(script, description)
            results[script] = success
        else:
            print(f"SKIP: {script} not found")
            results[script] = None

    # Generate report
    print("\n" + "=" * 70)
    print("Generating summary report...")
    print("=" * 70)

    report = generate_report()

    # Save report
    report_file = 'results/TEST_RESULTS.md'
    with open(report_file, 'w') as f:
        f.write(report)

    print(f"\n✓ Report saved: {report_file}\n")

    # Print summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for script, success in results.items():
        if success is None:
            status = "SKIPPED"
        elif success:
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
        print(f"{script:30s} : {status}")

    print("=" * 70)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    print("\nCheck the following:")
    print(f"  - Test report: {report_file}")
    print("  - Output files: results/world/ and results/psola/")
    print("  - Listen to converted audio files")

    # Return exit code
    if all(v in [True, None] for v in results.values()):
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
