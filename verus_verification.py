"""
Verus Verification Module

This module provides functions to verify generated Verus code and categorize errors.
"""

import subprocess
import re
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Optional


def extract_code_from_markdown(text: str) -> str:
    """
    Extract Verus code from markdown code blocks.

    Args:
        text: Generated text that may contain markdown code blocks

    Returns:
        Extracted code string
    """
    # Try to find ```verus ... ``` blocks
    verus_pattern = r"```verus\s*\n(.*?)```"
    matches = re.findall(verus_pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()

    # Try to find ```rust ... ``` blocks
    rust_pattern = r"```rust\s*\n(.*?)```"
    matches = re.findall(rust_pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()

    # Try generic ``` blocks
    generic_pattern = r"```\s*\n(.*?)```"
    matches = re.findall(generic_pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()

    # If no code blocks, return the text as-is
    return text.strip()


def categorize_verus_error(error_output: str) -> str:
    """
    Categorize Verus error based on error message patterns.

    Args:
        error_output: stderr output from Verus

    Returns:
        Error category string
    """
    error_lower = error_output.lower()

    # Syntax errors (Rust parsing)
    if "error: expected" in error_lower or "error: unexpected" in error_lower:
        return "syntax_error"
    if "parse error" in error_lower or "parsing failed" in error_lower:
        return "syntax_error"

    # Mode errors (exec/ghost/proof)
    if "mode" in error_lower and ("exec" in error_lower or "ghost" in error_lower or "proof" in error_lower):
        return "mode_error"
    if "cannot call ghost" in error_lower or "cannot call proof" in error_lower:
        return "mode_error"

    # Specification errors
    if "precondition not satisfied" in error_lower or "requires" in error_lower:
        return "precondition_error"
    if "postcondition not satisfied" in error_lower or "ensures" in error_lower:
        return "postcondition_error"
    if "invariant" in error_lower:
        return "invariant_error"

    # Termination errors
    if "decreases" in error_lower or "termination" in error_lower:
        return "termination_error"

    # Type errors
    if "type mismatch" in error_lower or "mismatched types" in error_lower:
        return "type_error"
    if "type error" in error_lower:
        return "type_error"

    # Verification condition failures
    if "assertion" in error_lower and ("not hold" in error_lower or "might fail" in error_lower):
        return "vc_failure"
    if "proof" in error_lower and "failed" in error_lower:
        return "vc_failure"

    # Timeout
    if "timeout" in error_lower or "timed out" in error_lower:
        return "timeout"

    # Unknown error
    return "unknown_error"


def verify_with_verus(
    code: str,
    verus_path: str = "verus",
    timeout: int = 30,
    temp_dir: Optional[str] = None,
) -> Dict:
    """
    Verify Verus code by running the Verus verifier.

    Args:
        code: Verus code to verify (can include markdown)
        verus_path: Path to Verus executable (default: "verus" in PATH)
        timeout: Timeout in seconds (default: 30)
        temp_dir: Directory for temporary files (default: system temp)

    Returns:
        Dictionary with verification results:
        {
            "success": bool,
            "errors": list[str],
            "error_type": str or None,
            "output": str,
            "code_extracted": str,
        }
    """
    # Extract code from markdown if present
    extracted_code = extract_code_from_markdown(code)

    # Wrap in minimal Verus boilerplate if needed
    if not extracted_code.startswith("use ") and "verus!" not in extracted_code:
        wrapped_code = f"""use vstd::prelude::*;

verus! {{

{extracted_code}

}} // verus!
"""
    else:
        wrapped_code = extracted_code

    # Create temporary file
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()

    temp_file = os.path.join(temp_dir, "verus_test.rs")

    try:
        # Write code to file
        with open(temp_file, "w") as f:
            f.write(wrapped_code)

        # Check if Verus is available
        check_result = subprocess.run(
            [verus_path, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if check_result.returncode != 0:
            return {
                "success": False,
                "errors": ["Verus not found or not executable"],
                "error_type": "verus_not_found",
                "output": "",
                "code_extracted": extracted_code,
            }

        # Run Verus verification
        result = subprocess.run(
            [verus_path, temp_file],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        # Parse results
        if result.returncode == 0:
            return {
                "success": True,
                "errors": [],
                "error_type": None,
                "output": result.stdout,
                "code_extracted": extracted_code,
            }
        else:
            error_lines = result.stderr.strip().split("\n")
            error_type = categorize_verus_error(result.stderr)

            return {
                "success": False,
                "errors": error_lines,
                "error_type": error_type,
                "output": result.stderr,
                "code_extracted": extracted_code,
            }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "errors": [f"Verification timed out after {timeout} seconds"],
            "error_type": "timeout",
            "output": "",
            "code_extracted": extracted_code,
        }

    except FileNotFoundError:
        return {
            "success": False,
            "errors": [f"Verus executable not found at: {verus_path}"],
            "error_type": "verus_not_found",
            "output": "",
            "code_extracted": extracted_code,
        }

    except Exception as e:
        return {
            "success": False,
            "errors": [f"Unexpected error: {str(e)}"],
            "error_type": "unknown_error",
            "output": "",
            "code_extracted": extracted_code,
        }

    finally:
        # Clean up temporary file
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass  # Ignore cleanup errors


def batch_verify(
    code_samples: List[str],
    verus_path: str = "verus",
    timeout: int = 30,
) -> List[Dict]:
    """
    Verify multiple code samples in batch.

    Args:
        code_samples: List of Verus code strings to verify
        verus_path: Path to Verus executable
        timeout: Timeout per sample in seconds

    Returns:
        List of verification result dictionaries
    """
    results = []
    for i, code in enumerate(code_samples):
        print(f"Verifying sample {i+1}/{len(code_samples)}...", end=" ")
        result = verify_with_verus(code, verus_path, timeout)
        results.append(result)
        status = "✓" if result["success"] else f"✗ ({result['error_type']})"
        print(status)

    return results


def compute_verification_metrics(results: List[Dict]) -> Dict:
    """
    Compute aggregate metrics from verification results.

    Args:
        results: List of verification result dictionaries

    Returns:
        Dictionary with aggregate metrics
    """
    total = len(results)
    if total == 0:
        return {"error": "No results provided"}

    successful = sum(1 for r in results if r["success"])
    failed = total - successful

    # Count error types
    error_counts = {}
    for r in results:
        if not r["success"]:
            error_type = r["error_type"]
            error_counts[error_type] = error_counts.get(error_type, 0) + 1

    return {
        "total_samples": total,
        "successful": successful,
        "failed": failed,
        "verification_pass_rate": successful / total,
        "error_counts": error_counts,
        "error_breakdown": {
            "syntax_errors": error_counts.get("syntax_error", 0),
            "mode_errors": error_counts.get("mode_error", 0),
            "precondition_errors": error_counts.get("precondition_error", 0),
            "postcondition_errors": error_counts.get("postcondition_error", 0),
            "invariant_errors": error_counts.get("invariant_error", 0),
            "termination_errors": error_counts.get("termination_error", 0),
            "type_errors": error_counts.get("type_error", 0),
            "vc_failures": error_counts.get("vc_failure", 0),
            "timeouts": error_counts.get("timeout", 0),
            "verus_not_found": error_counts.get("verus_not_found", 0),
            "unknown_errors": error_counts.get("unknown_error", 0),
        },
    }


if __name__ == "__main__":
    # Test the verification module
    print("Testing Verus verification module...\n")

    # Test 1: Valid Verus code
    test_code_valid = """
fn abs(x: i32) -> i32
    requires x != i32::MIN;
    ensures result >= 0;
    ensures result == x || result == -x;
{
    if x < 0 { -x } else { x }
}
"""

    # Test 2: Invalid code (missing precondition)
    test_code_invalid = """
fn abs(x: i32) -> i32
    ensures result >= 0;
{
    if x < 0 { -x } else { x }
}
"""

    # Test 3: Code in markdown
    test_code_markdown = """```verus
fn max(a: i32, b: i32) -> i32
    ensures result >= a && result >= b;
{
    if a > b { a } else { b }
}
```"""

    print("Test 1: Valid code")
    result1 = verify_with_verus(test_code_valid)
    print(f"  Success: {result1['success']}")
    print(f"  Error type: {result1['error_type']}")

    print("\nTest 2: Invalid code (missing precondition)")
    result2 = verify_with_verus(test_code_invalid)
    print(f"  Success: {result2['success']}")
    print(f"  Error type: {result2['error_type']}")

    print("\nTest 3: Code in markdown")
    result3 = verify_with_verus(test_code_markdown)
    print(f"  Success: {result3['success']}")
    print(f"  Code extracted: {len(result3['code_extracted'])} chars")

    print("\nComputing metrics...")
    metrics = compute_verification_metrics([result1, result2, result3])
    print(f"  Pass rate: {metrics['verification_pass_rate']:.1%}")
    print(f"  Error breakdown: {metrics['error_breakdown']}")
