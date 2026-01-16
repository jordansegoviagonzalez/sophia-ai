import sys
import os
from datasets import load_from_disk
from rich.console import Console
from rich.panel import Panel

console = Console()

# Correct Path based on your verification
EXTERNAL_DATA_PATH = "../../../AI/ai-ml-backend-datasets/processed/ai-ml-backend-knowledge-v3"

def inspect():
    # Resolve to absolute path to be safe
    abs_path = os.path.abspath(EXTERNAL_DATA_PATH)
    console.print(f"[bold blue]Inspecting External Data:[/bold blue] {abs_path}")
    
    if not os.path.exists(abs_path):
        console.print(f"[bold red]Error:[/bold red] Path does not exist: {abs_path}")
        return

    try:
        # Load train split
        ds = load_from_disk(os.path.join(abs_path, "train"))
        console.print(f"Loaded 'train' split. Size: [green]{len(ds)}[/green] rows.")
        
        # Show first 3 examples
        for i in range(min(3, len(ds))):
            example = ds[i]
            # Print keys to confirm schema
            console.print(f"[dim]Keys: {list(example.keys())}[/dim]")
            
            q = example.get('instruction', example.get('question', 'N/A'))
            a = example.get('output', example.get('response', 'N/A'))
            
            console.print(Panel(
                f"[bold]Q:[/bold] {q}\n\n[bold]A:[/bold] {a[:300]}...",
                title=f"Example {i+1}",
                border_style="cyan"
            ))
            
    except Exception as e:
        console.print(f"[bold red]Failed to load dataset:[/bold red] {e}")

if __name__ == "__main__":
    inspect()
