#!/usr/bin/env python3

import os
import subprocess
import argparse
from pathlib import Path
from typing import List
from pathlib import Path


def vendor_notebook(original_nb: str, modules_dir: str) -> str:
    orig_path = Path(original_nb)
    tmp_nb = orig_path.with_suffix(".vendor.py")

    header_lines = [
        "# THIS FILE IS AUTO‑GENERATED — includes your local helpers",
        "# Do not edit directly; see scripts/build.py for the source merge logic",
    ]

    module_code = []
    for pyfile in Path(modules_dir).rglob("*.py"):
        rel = pyfile.relative_to(Path(modules_dir))
        module_code.append(f"\n# --- MODULE: {rel} ---\n")
        module_code.append(pyfile.read_text())

    # Write out the merged notebook
    tmp_nb.write_text(
        "\n".join(header_lines)
        + "\n\n"
        + "".join(module_code)
        + "\n\n"
        + orig_path.read_text()
    )

    return str(tmp_nb)


def export_html_wasm(notebook_path: str, output_dir: str, as_app: bool = False) -> bool:
    output_path = notebook_path.replace(".py", ".html")
    nb_to_export = vendor_notebook(notebook_path, modules_dir="src")

    cmd = ["marimo", "export", "html-wasm", nb_to_export]

    if as_app:
        print(f"Exporting {notebook_path} to {output_path} as app")
        cmd += ["--mode", "run", "--no-show-code"]
    else:
        print(f"Exporting {notebook_path} to {output_path} as notebook")
        cmd += ["--mode", "edit"]

    output_file = os.path.join(output_dir, output_path)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    cmd += ["-o", output_file]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error exporting {notebook_path}:")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"Unexpected error exporting {notebook_path}: {e}")
        return False


def generate_index(all_notebooks: List[str], output_dir: str) -> None:
    """Generate the index.html file."""
    print("Generating index.html")

    index_path = os.path.join(output_dir, "index.html")
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(index_path, "w") as f:
            f.write(
                """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>marimo</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
  </head>
  <body class="font-sans max-w-2xl mx-auto p-8 leading-relaxed">
    <div class="mb-8">
      <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-thick.svg" alt="marimo" class="h-20" />
    </div>
    <div class="grid gap-4">
"""
            )
            for notebook in all_notebooks:
                notebook_name = notebook.split("/")[-1].replace(".py", "")
                display_name = notebook_name.replace("_", " ").title()

                f.write(
                    f'      <div class="p-4 border border-gray-200 rounded">\n'
                    f'        <h3 class="text-lg font-semibold mb-2">{display_name}</h3>\n'
                    f'        <div class="flex gap-2">\n'
                    f'          <a href="{notebook.replace(".py", ".html")}" class="px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded">Open Notebook</a>\n'
                    f"        </div>\n"
                    f"      </div>\n"
                )
            f.write(
                """    </div>
  </body>
</html>"""
            )
    except IOError as e:
        print(f"Error generating index.html: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build marimo notebooks")
    parser.add_argument(
        "--output-dir", default="_site", help="Output directory for built files"
    )
    args = parser.parse_args()

    all_notebooks: List[str] = []
    for directory in ["notebooks"]:
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"Warning: Directory not found: {dir_path}")
            continue

        all_notebooks.extend(str(path) for path in dir_path.rglob("*.py"))

    if not all_notebooks:
        print("No notebooks found!")
        return

    # Export notebooks sequentially
    for nb in all_notebooks:
        export_html_wasm(nb, args.output_dir, as_app=nb.startswith("apps/"))

    # Generate index only if all exports succeeded
    generate_index(all_notebooks, args.output_dir)


if __name__ == "__main__":
    main()
