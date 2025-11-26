import os
import re
import glob
import shutil
import subprocess
import PyPDF2


def compress_pdf_ghostscript(
    input_pdf: str, output_pdf: str, quality: str = "/ebook"
) -> None:
    """Compress PDF size."""
    cmd = [
        "gs",
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        f"-dPDFSETTINGS={quality}",
        "-dNOPAUSE",
        "-dQUIET",
        "-dBATCH",
        f"-sOutputFile={output_pdf}",
        input_pdf,
    ]
    subprocess.run(cmd, check=True)
    print(f"Compressed PDF saved to {output_pdf}")


def find_main_tex_file_to_combine(folder: str) -> str:
    """
    Find the main text file in a latex source folder,
    return that main file.
    """
    tex_files = glob.glob(os.path.join(folder, "**/*.tex"), recursive=True)
    for tex_file in tex_files:
        with open(tex_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            if "author" in content and "\\documentclass" in content:
                return tex_file
    # fallback: return largest
    if tex_files:
        return max(tex_files, key=os.path.getsize)
    else:
        raise Exception("folder doesn't contain tex files")


def find_file(filename: str, base_path: str, max_up: int = 1) -> str | None:
    """
    Find file in a folder,
    return found candidate files or None.
    """
    for i in range(max_up + 1):
        candidate = os.path.join(base_path, *([".."] * i), filename)
        candidate = os.path.realpath(candidate)
        if os.path.exists(candidate):
            return candidate
    return None


def resolve_inputs(content: str, base_path: str, visited: set | None = None) -> str:
    """
    Replace every input{} and include{} in the latex file to combine everything into 1 file,
    return the combined source.
    """
    if visited is None:
        visited = set()

    # regex for finding \input{} and \include{}
    pattern = re.compile(r"(?m)^[^%\n]*\\(input|include){([^}]+)}")

    # function to iteratively replace all \input and \include
    def replacer(match):
        cmd, filename = match.groups()
        filepath = os.path.join(base_path, filename)
        if not filepath.endswith(".tex"):
            filepath += ".tex"

        # Avoid circular includes
        realpath = find_file(
            filename if filename.endswith(".tex") else filename + ".tex", base_path
        )
        if not realpath or realpath in visited or not os.path.exists(realpath):
            return f"% Skipped {cmd}{{{filename}}} (already included or not found)"

        visited.add(realpath)
        try:
            with open(realpath, "r", encoding="utf-8", errors="ignore") as f:
                nested_content = f.read()
            return resolve_inputs(nested_content, os.path.dirname(realpath), visited)
        except Exception as e:
            return f"% Failed to include {cmd}{{{filename}}}: {e}"

    return pattern.sub(replacer, content)


def combine_latex_sources(paper_dir: str, output_file: str = "combined.tex") -> str:
    """
    Combine and save the combined latex source as a latex tex file,
    return the filename.
    """
    main_tex = find_main_tex_file_to_combine(paper_dir)

    with open(main_tex, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # remove any comments
    processed_lines = []
    for line in content.splitlines():
        # skip lines starting with % (comments)
        if line.lstrip().startswith("%"):
            continue
        processed_lines.append(line)

    no_comments = "\n".join(processed_lines)
    # combine into one main file if there are inputs
    combined_content = resolve_inputs(no_comments, os.path.dirname(main_tex))

    combined_path = os.path.join(paper_dir, output_file)
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write(combined_content)
    return combined_path


def copy_dir_contents(src: str, dst: str, new_name: str | None = None) -> None:
    """
    Copy all files and subdirectories from src to dst,
    return None.
    """
    if not os.path.exists(src):
        print("ERROR: Source directory does not exist.")
        return

    file_count = 0
    skipped_count = 0
    for root, _, files in os.walk(src):
        rel_path = os.path.relpath(root, src)
        dest_dir = os.path.join(dst, rel_path)
        os.makedirs(dest_dir, exist_ok=True)

        for file in files:
            src_file = os.path.join(root, file)

            # Only rename files exactly named 'main.tex'
            filename_no_ext, ext = os.path.splitext(file)
            if new_name and filename_no_ext == "main" and ext == "tex":
                dest_file_name = new_name + ext
            else:
                dest_file_name = file

            dst_file = os.path.join(dest_dir, dest_file_name)
            if os.path.exists(dst_file):
                skipped_count += 1
            else:
                shutil.copy2(src_file, dst_file)
                file_count += 1
    print(f"Copied {file_count} files, skipped {skipped_count} files.")


def compile_latex(paper: str, des: str, main_tex: str) -> str:
    """
    Compile a LaTeX project into a PDF,
    return output pdf path.
    """
    dst = des.lstrip("/\\")
    folder = os.getcwd()
    des = os.path.join(folder, dst)

    main_tex_filename = os.path.basename(main_tex)
    base_name = os.path.splitext(main_tex_filename)[0]
    copy_dir_contents(f"data/papers/{paper}", dst, base_name)
    output_pdf_path = os.path.join(des, main_tex_filename.replace(".tex", ".pdf"))

    # remove previous pdf
    if os.path.exists(output_pdf_path):
        os.remove(output_pdf_path)

    latex_cmd = [
        "/Library/TeX/texbin/pdflatex",
        "-interaction=nonstopmode",
        "--shell-escape",
        "-output-directory",
        des,
        main_tex_filename,
    ]

    bibtex_cmd = ["/Library/TeX/texbin/bibtex", base_name]

    # try to compile twice, replace references if first time doesn't work
    for attempt in range(2):
        try:
            with open("data/latex_compilation_log.txt", "w") as log:
                # 1st run: generate .aux
                print("start")
                subprocess.run(
                    latex_cmd,
                    cwd=os.path.dirname(main_tex),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=60,
                )
                print("finished 1")

                # 2nd run: process bibliography
                subprocess.run(
                    bibtex_cmd,
                    cwd=des,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                print("finished 2")

                # 3rd and 4th runs: resolve refs
                subprocess.run(
                    latex_cmd,
                    cwd=os.path.dirname(main_tex),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=60,
                )
                print("finished 3")
                subprocess.run(
                    latex_cmd,
                    cwd=os.path.dirname(main_tex),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=60,
                )
                print("finished 4")

            print(f"Compilation SUCCESSFUL: {output_pdf_path}")
            break
        except subprocess.CalledProcessError:
            print("Compilation FAILED! Check latex_compilation_log.txt for details.")
            replace_bibliography(main_tex)

    return output_pdf_path


def replace_bibliography(filepath: str) -> None:
    """
    Replace bibliography{*.bib} with input{*.bbl} since there could be no bib files,
    so that the references can compile.
    """
    try:
        # Read the entire content of the file
        with open(filepath, "r", encoding="utf-8") as file:
            content = file.read()

        pattern = r"\\bibliography\{.*?\}"
        replacement = r"\\input{main.bbl}"
        new_content = re.sub(
            pattern, replacement, content, flags=re.IGNORECASE | re.DOTALL
        )

        with open(filepath, "w", encoding="utf-8") as file:
            file.write(new_content)
        print(
            f"Successfully replaced \\bibliography{{...}} with \\input{{main.bbl}} in {filepath}."
        )

    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def check_pdf_for_unresolved_references(pdf_file_path: str) -> bool:
    """
    Perform OCR to check if there are unresolved references,
    return whether we should use the pdf or not (should not if
    there are unresolved references).
    """
    try:
        with open(pdf_file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text = page.extract_text()
                # assume there will be a reference on the first 5 pages
                if ("???" in text or "[?]" in text) and (page_num + 1 < 5):
                    print(f"Unresolved referesnces found on page {page_num + 1}.")
                    return False
            return True
    except FileNotFoundError:
        print(f"Error: {pdf_file_path} not found.")
        return False
    except Exception as e:
        print(f"Error reading {pdf_file_path}: {e}")
        return False
