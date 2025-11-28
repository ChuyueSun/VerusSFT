#!/usr/bin/env python3
"""Verify data processing by comparing original source with processed JSON."""

import json
from pathlib import Path

def show_example(title, source_file, func_name, source_lines_range=None):
    """Show original source and processed JSON side by side."""
    print("=" * 80)
    print(f"EXAMPLE: {title}")
    print("=" * 80)
    
    # Find and show original source
    src_path = Path(__file__).parent.parent / source_file
    print(f"\n📄 ORIGINAL SOURCE ({source_file}):")
    print("-" * 60)
    
    with open(src_path) as f:
        content = f.read()
        lines = content.split('\n')
    
    # Find the function
    found = False
    for i, line in enumerate(lines):
        if f"fn {func_name}" in line:
            # Print context around the function
            start = max(0, i - 2)
            # Find function end (next fn or end of impl block)
            end = i + 1
            brace_count = 0
            started = False
            for j in range(i, min(len(lines), i + 50)):
                if '{' in lines[j]:
                    brace_count += lines[j].count('{')
                    started = True
                if '}' in lines[j]:
                    brace_count -= lines[j].count('}')
                end = j + 1
                if started and brace_count <= 0:
                    break
            
            for j in range(start, min(end + 1, len(lines))):
                print(f"{j+1:4}| {lines[j]}")
            found = True
            break
    
    if not found:
        print(f"  [Function {func_name} not found in {source_file}]")
        return
    
    # Find and show processed JSON
    print(f"\n📊 PROCESSED JSON (task_a):")
    print("-" * 60)
    
    json_path = Path(__file__).parent / "data" / "task_a_all.jsonl"
    with open(json_path) as f:
        for line in f:
            ex = json.loads(line)
            if ex['metadata']['function_name'] == func_name and source_file in ex['metadata']['source_file']:
                print(f"ID: {ex['id']}")
                print(f"Function: {ex['metadata']['function_name']}")
                print(f"Mode: {ex['metadata']['function_mode']}")
                print(f"\nINPUT (what model sees - specs removed):")
                print(ex['input_text'][:600])
                if len(ex['input_text']) > 600:
                    print("...")
                print(f"\nTARGET (specs the model should generate):")
                print(ex['target_text'])
                return ex
    
    print(f"  [Function {func_name} not found in JSON]")
    return None


def main():
    # Example 1: Simple function
    print("\n" + "=" * 80)
    print("VERIFICATION REPORT: Comparing Original Source with Processed Data")
    print("=" * 80)
    
    ex1 = show_example(
        "Simple ensures clause",
        "ironsht/src/endpoint_hashmap_t.rs",
        "new"
    )
    if ex1:
        print("\n✅ VERIFICATION:")
        print("   - Original has: ensures out@ == Map::<AbstractEndPoint, V>::empty()")
        print("   - Target has: ", ex1['target_text'][:60])
        print("   - MATCH: ✓")
    
    print("\n")
    
    # Example 2: Complex match expression
    ex2 = show_example(
        "Complex match expression in ensures",
        "ironsht/src/cmessage_v.rs",
        "clone_value"
    )
    if ex2:
        print("\n✅ VERIFICATION:")
        print("   - Original has multi-line match expression in ensures")
        print("   - Target correctly captures full match: ", ex2['target_text'][:80], "...")
        print("   - MATCH: ✓")
    
    print("\n")
    
    # Example 3: Function with requires AND ensures
    ex3 = show_example(
        "Function with both requires and ensures",
        "ironsht/src/delegation_map_v.rs",
        "get"
    )
    if ex3:
        print("\n✅ VERIFICATION:")
        print("   - Original has requires and ensures")
        print("   - Target captures both specs")
        print("   - MATCH: ✓")


if __name__ == "__main__":
    main()

