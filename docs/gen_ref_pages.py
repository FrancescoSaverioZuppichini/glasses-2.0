"""Generate the code reference pages and navigation."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

src_dir = "glasses"
exclude = "tests"

for path in sorted(Path(src_dir).rglob("*.py")):
    if exclude not in str(path):
        module_path = path.relative_to(src_dir).with_suffix("")
        doc_path = path.relative_to(src_dir).with_suffix(".md")
        full_doc_path = Path("reference", doc_path)

        # Handle edge cases
        parts = (src_dir,) + tuple(module_path.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
            doc_path = doc_path.with_name("index.md")
            full_doc_path = full_doc_path.with_name("index.md")
        elif parts[-1] == "__main__":
            continue

        nav[parts] = doc_path.as_posix()  #

        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            ident = ".".join(parts)
            to_write = f"::: {ident}"
            to_write += f"\n    rendering:"
            to_write += f"\n        show_if_no_docstring: false"
            fd.write(to_write)

        mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:  #
    nav_file.writelines(nav.build_literate_nav())  #
