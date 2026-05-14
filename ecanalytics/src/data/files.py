import os

ALLOWED_FILE_EXTENSIONS = {".txt", ".csv"}

def is_allowed_file(file_path: str, extensions: set[str] | None = None) -> bool:
    if extensions is None:
        extensions = ALLOWED_FILE_EXTENSIONS

    return (
        os.path.isfile(file_path) and os.path.splitext(file_path)[1] in extensions
    )


def parse_file_name(
    file_path: str, require_sample_no: bool = True
) -> tuple[int, list[str]] | None:
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    name_parts = file_name.split("_")

    date_part = name_parts[0]
    if len(date_part) != 8 or not date_part.isdigit():
        return None

    if require_sample_no:
        for i, part in enumerate(name_parts[1:], start=1):
            if part.startswith("S") and part[1:].isdigit():
                sample_number = int(name_parts.pop(i)[1:])
                return (sample_number, name_parts + [f"S{sample_number:02d}"])

        return None

    return 1, name_parts


def files_from_folder(
    folder_path: str, extensions: set[str] | None = None
) -> list[str]:
    if extensions is None:
        extensions = ALLOWED_FILE_EXTENSIONS

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    files = []
    for item in os.listdir(folder_path):
        path = os.path.join(folder_path, item)
        if is_allowed_file(path, extensions):
            files.append(path)

    return files
