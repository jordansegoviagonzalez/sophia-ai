import sys
import re
from datasets import load_from_disk
from collections import Counter
from rich.console import Console
from rich.table import Table

console = Console()

DATA_PATH = "../../ai-ml-backend-datasets/processed/ai-ml-backend-knowledge-v1"

def check_pii(text):
    """
    Simple regex check for PII (Emails, IPs).
    Enterprise grade would use Presidio, but this proves the concept.
    """
    if not text: return False
    # Regex for Email
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    # Regex for IPv4
    ip_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
    
    if re.search(email_pattern, text) or re.search(ip_pattern, text):
        return True
    return False

def audit():
    console.print(f"[bold blue]Starting Enterprise Data Audit on:[/bold blue] {DATA_PATH}")
    
    try:
        train_ds = load_from_disk(f"{DATA_PATH}/train")
        val_ds = load_from_disk(f"{DATA_PATH}/validation")
    except Exception as e:
        console.print(f"[bold red]FATAL:[/bold red] Could not load datasets. {e}")
        sys.exit(1)

    # 1. Volume Check
    console.print(f"Training Samples: [green]{len(train_ds)}[/green]")
    console.print(f"Validation Samples: [green]{len(val_ds)}[/green]")

    # 2. Null/Empty Check
    issues = 0
    for split_name, ds in [("Train", train_ds), ("Val", val_ds)]:
        for i, row in enumerate(ds):
            # Check output (the most critical part)
            if not row['output'] or len(row['output'].strip()) < 10:
                console.print(f"[red]WARNING:[/red] {split_name} row {i} has empty/short output.")
                issues += 1
            # Check instruction
            if not row['instruction']:
                console.print(f"[red]WARNING:[/red] {split_name} row {i} has missing instruction.")
                issues += 1

    # 3. Deduplication Check
    # We check if 'instruction' + 'input' is unique
    seen = set()
    dupes = 0
    for row in train_ds:
        sig = (row['instruction'] or "") + (row['input'] or "")
        if sig in seen:
            dupes += 1
        seen.add(sig)
    
    if dupes > 0:
        console.print(f"[yellow]WARNING:[/yellow] Found {dupes} duplicate training prompts. (Overfitting Risk)")
    else:
        console.print("[green]PASS:[/green] No exact duplicates found.")

    # 4. PII Scan
    pii_count = 0
    for row in train_ds:
        if check_pii(row['output']) or check_pii(row['input']):
            pii_count += 1
            
    if pii_count > 0:
        console.print(f"[yellow]WARNING:[/yellow] Potential PII (Email/IP) found in {pii_count} rows.")
    else:
        console.print("[green]PASS:[/green] No PII patterns detected.")

    # 5. Schema Validation
    expected_cols = {'instruction', 'input', 'output', 'source', 'category'}
    found_cols = set(train_ds.column_names)
    if expected_cols.issubset(found_cols):
        console.print("[green]PASS:[/green] Schema is compliant (Governance Tags Found).")
    else:
        console.print(f"[red]FAIL:[/red] Schema missing columns: {expected_cols - found_cols}")

    console.print("\n[bold]Audit Complete.[/bold]")

if __name__ == "__main__":
    audit()
