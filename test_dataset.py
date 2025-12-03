"""
Test Dataset for VerusSFT Evaluation

IMPORTANT: These examples are HELD-OUT test data and must NEVER be used for training.
They are used exclusively for evaluating baseline and fine-tuned models.

The training examples are in sft_example.py:build_dataset()
"""

from typing import List, Dict


def build_test_dataset() -> List[Dict[str, str]]:
    """
    Build held-out test examples for evaluation.

    These examples are DIFFERENT from training examples and cover:
    - Specification generation (Task A)
    - Code synthesis (Task B)
    - Various Verus patterns (arrays, loops, ghost code, Views)

    Returns:
        List of test examples with 'prompt', 'expected_output', and 'task_type' fields.
    """
    test_examples = [
        # === Specification Generation Examples (Task A) ===
        {
            "prompt": "Add Verus specs to this clamp function:\n```rust\nfn clamp(x: i32, min: i32, max: i32) -> i32 {\n    if x < min { min } else if x > max { max } else { x }\n}\n```",
            "expected_output": """```verus
fn clamp(x: i32, min: i32, max: i32) -> i32 {
    requires min <= max;
    ensures result >= min && result <= max;
    ensures (x >= min && x <= max) ==> result == x;
    ensures x < min ==> result == min;
    ensures x > max ==> result == max;
    if x < min { min } else if x > max { max } else { x }
}
```""",
            "task_type": "spec_generation",
        },
        {
            "prompt": "Add Verus specs for this swap function:\n```rust\nfn swap(a: &mut i32, b: &mut i32) {\n    let temp = *a;\n    *a = *b;\n    *b = temp;\n}\n```",
            "expected_output": """```verus
fn swap(a: &mut i32, b: &mut i32)
    ensures *a == old(*b);
    ensures *b == old(*a);
{
    let temp = *a;
    *a = *b;
    *b = temp;
}
```""",
            "task_type": "spec_generation",
        },
        {
            "prompt": "Add Verus specs for safe vector indexing:\n```rust\nfn get_or_default(v: &Vec<i32>, idx: usize, default: i32) -> i32 {\n    if idx < v.len() {\n        v[idx]\n    } else {\n        default\n    }\n}\n```",
            "expected_output": """```verus
fn get_or_default(v: &Vec<i32>, idx: usize, default: i32) -> i32
    ensures idx < v.len() ==> result == v[idx as int];
    ensures idx >= v.len() ==> result == default;
{
    if idx < v.len() {
        v[idx]
    } else {
        default
    }
}
```""",
            "task_type": "spec_generation",
        },
        {
            "prompt": "Add Verus specs for this modulo function:\n```rust\nfn safe_mod(a: u32, b: u32) -> u32 {\n    a % b\n}\n```",
            "expected_output": """```verus
fn safe_mod(a: u32, b: u32) -> u32
    requires b > 0;
    ensures result < b;
    ensures result == a % b;
{
    a % b
}
```""",
            "task_type": "spec_generation",
        },
        {
            "prompt": "Add Verus specs for increment:\n```rust\nfn increment(x: &mut u32) {\n    *x = *x + 1;\n}\n```",
            "expected_output": """```verus
fn increment(x: &mut u32)
    requires *x < u32::MAX;
    ensures *x == old(*x) + 1;
{
    *x = *x + 1;
}
```""",
            "task_type": "spec_generation",
        },

        # === Code Synthesis Examples (Task B) ===
        {
            "prompt": "Write a Verus function that computes the average of two numbers (rounded down):",
            "expected_output": """```verus
fn average(a: u32, b: u32) -> u32
    ensures result <= a && result <= b || result >= a && result >= b;
    ensures result == (a + b) / 2;
{
    (a + b) / 2
}
```""",
            "task_type": "code_synthesis",
        },
        {
            "prompt": "Write a Verus function that returns the maximum of three numbers:",
            "expected_output": """```verus
fn max3(a: i32, b: i32, c: i32) -> i32
    ensures result >= a && result >= b && result >= c;
    ensures result == a || result == b || result == c;
{
    if a >= b && a >= c {
        a
    } else if b >= c {
        b
    } else {
        c
    }
}
```""",
            "task_type": "code_synthesis",
        },
        {
            "prompt": "Write a Verus function that checks if a number is even:",
            "expected_output": """```verus
fn is_even(x: i32) -> bool
    ensures result == (x % 2 == 0);
{
    x % 2 == 0
}
```""",
            "task_type": "code_synthesis",
        },
        {
            "prompt": "Write a Verus function for safe subtraction that returns 0 if result would be negative:",
            "expected_output": """```verus
fn saturating_sub(a: u32, b: u32) -> u32
    ensures a >= b ==> result == a - b;
    ensures a < b ==> result == 0;
    ensures result <= a;
{
    if a >= b {
        a - b
    } else {
        0
    }
}
```""",
            "task_type": "code_synthesis",
        },
        {
            "prompt": "Write a Verus function that computes the absolute difference between two numbers:",
            "expected_output": """```verus
fn abs_diff(a: i32, b: i32) -> i32
    requires a != i32::MIN || b >= 0;
    requires b != i32::MIN || a >= 0;
    ensures result >= 0;
    ensures result == a - b || result == b - a;
{
    if a >= b {
        a - b
    } else {
        b - a
    }
}
```""",
            "task_type": "code_synthesis",
        },

        # === Array/Vector Examples ===
        {
            "prompt": "Add Verus specs for finding the last element:\n```rust\nfn get_last(arr: &[i32]) -> Option<i32> {\n    if arr.len() > 0 {\n        Some(arr[arr.len() - 1])\n    } else {\n        None\n    }\n}\n```",
            "expected_output": """```verus
fn get_last(arr: &[i32]) -> Option<i32>
    ensures arr.len() > 0 ==> result == Some(arr[arr.len() - 1]);
    ensures arr.len() == 0 ==> result == None;
{
    if arr.len() > 0 {
        Some(arr[arr.len() - 1])
    } else {
        None
    }
}
```""",
            "task_type": "spec_generation",
        },
        {
            "prompt": "Write a Verus function that checks if a vector is empty:",
            "expected_output": """```verus
fn is_empty(v: &Vec<i32>) -> bool
    ensures result == (v.len() == 0);
{
    v.len() == 0
}
```""",
            "task_type": "code_synthesis",
        },

        # === Loop Examples ===
        {
            "prompt": "Add Verus specs with loop invariant for sum:\n```rust\nfn sum_n(n: u32) -> u32 {\n    let mut sum = 0;\n    let mut i = 0;\n    while i <= n {\n        sum = sum + i;\n        i = i + 1;\n    }\n    sum\n}\n```",
            "expected_output": """```verus
fn sum_n(n: u32) -> u32
    requires n <= 1000;  // Prevent overflow
    ensures result == n * (n + 1) / 2;
{
    let mut sum = 0;
    let mut i = 0;
    while i <= n
        invariant i <= n + 1;
        invariant sum == i * (i - 1) / 2;
    {
        sum = sum + i;
        i = i + 1;
    }
    sum
}
```""",
            "task_type": "spec_generation",
        },

        # === Comparison/Ordering Examples ===
        {
            "prompt": "Write a Verus function that checks if three numbers are in ascending order:",
            "expected_output": """```verus
fn is_ascending(a: i32, b: i32, c: i32) -> bool
    ensures result == (a <= b && b <= c);
{
    a <= b && b <= c
}
```""",
            "task_type": "code_synthesis",
        },
        {
            "prompt": "Add Verus specs for range checking:\n```rust\nfn in_range(x: i32, low: i32, high: i32) -> bool {\n    x >= low && x <= high\n}\n```",
            "expected_output": """```verus
fn in_range(x: i32, low: i32, high: i32) -> bool
    ensures result == (x >= low && x <= high);
{
    x >= low && x <= high
}
```""",
            "task_type": "spec_generation",
        },

        # === Bitwise/Arithmetic Examples ===
        {
            "prompt": "Write a Verus function that checks if a number is a power of 2:",
            "expected_output": """```verus
fn is_power_of_two(x: u32) -> bool
    ensures result ==> x > 0;
    ensures result == (x > 0 && (x & (x - 1)) == 0);
{
    x > 0 && (x & (x - 1)) == 0
}
```""",
            "task_type": "code_synthesis",
        },
        {
            "prompt": "Add Verus specs for multiplication with overflow check:\n```rust\nfn checked_mul(a: u32, b: u32) -> Option<u32> {\n    if a > 0 && b > u32::MAX / a {\n        None\n    } else {\n        Some(a * b)\n    }\n}\n```",
            "expected_output": """```verus
fn checked_mul(a: u32, b: u32) -> Option<u32>
    ensures a > 0 && b > u32::MAX / a ==> result == None;
    ensures !(a > 0 && b > u32::MAX / a) ==> result == Some(a * b);
{
    if a > 0 && b > u32::MAX / a {
        None
    } else {
        Some(a * b)
    }
}
```""",
            "task_type": "spec_generation",
        },

        # === Boolean Logic Examples ===
        {
            "prompt": "Write a Verus function for logical XOR:",
            "expected_output": """```verus
fn xor(a: bool, b: bool) -> bool
    ensures result == ((a || b) && !(a && b));
    ensures result == (a != b);
{
    a != b
}
```""",
            "task_type": "code_synthesis",
        },
        {
            "prompt": "Write a Verus function that checks if exactly one of two booleans is true:",
            "expected_output": """```verus
fn exactly_one(a: bool, b: bool) -> bool
    ensures result == ((a && !b) || (!a && b));
    ensures result == (a != b);
{
    (a && !b) || (!a && b)
}
```""",
            "task_type": "code_synthesis",
        },

        # === Sign/Comparison Examples ===
        {
            "prompt": "Write a Verus function that returns the sign of a number (-1, 0, or 1):",
            "expected_output": """```verus
fn sign(x: i32) -> i32
    ensures x > 0 ==> result == 1;
    ensures x == 0 ==> result == 0;
    ensures x < 0 ==> result == -1;
    ensures result >= -1 && result <= 1;
{
    if x > 0 {
        1
    } else if x < 0 {
        -1
    } else {
        0
    }
}
```""",
            "task_type": "code_synthesis",
        },
    ]

    return test_examples


def build_few_shot_examples() -> str:
    """
    Build few-shot examples for in-context learning.
    These are separate from both training and test sets.

    Returns:
        Formatted string with example prompt-completion pairs.
    """
    few_shot = """Here are some examples of Verus code generation:

Example 1:
Prompt: Add Verus specs to this function:
```rust
fn abs(x: i32) -> i32 {
    if x < 0 { -x } else { x }
}
```

Answer:
```verus
fn abs(x: i32) -> i32
    requires x != i32::MIN;
    ensures result >= 0;
    ensures result == x || result == -x;
{
    if x < 0 { -x } else { x }
}
```

Example 2:
Prompt: Write a Verus function that multiplies by 2:

Answer:
```verus
fn double(x: i32) -> i32
    requires x < i32::MAX / 2 && x > i32::MIN / 2;
    ensures result == 2 * x;
{
    x * 2
}
```

Example 3:
Prompt: Add Verus specs for this max function:
```rust
fn max(a: i32, b: i32) -> i32 {
    if a > b { a } else { b }
}
```

Answer:
```verus
fn max(a: i32, b: i32) -> i32
    ensures result >= a && result >= b;
    ensures result == a || result == b;
{
    if a > b { a } else { b }
}
```

Now solve this task:

"""
    return few_shot


if __name__ == "__main__":
    # Print test set statistics
    test_set = build_test_dataset()
    print(f"Test set size: {len(test_set)} examples")

    task_counts = {}
    for example in test_set:
        task_type = example["task_type"]
        task_counts[task_type] = task_counts.get(task_type, 0) + 1

    print("\nTask distribution:")
    for task_type, count in task_counts.items():
        print(f"  {task_type}: {count}")

    print(f"\nFirst example:")
    print(f"Prompt: {test_set[0]['prompt'][:100]}...")
    print(f"Expected: {test_set[0]['expected_output'][:100]}...")
