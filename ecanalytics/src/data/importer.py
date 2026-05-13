import os


class Importer:
    _ALLOWED_FILE_EXTENSIONS = {".txt", ".csv"}

    @classmethod
    def is_allowed_file(cls, file_path: str, extensions: set[str] | None = None) -> bool:
        """Checks if the file has an allowed extension."""
        if extensions is None:
            extensions = cls._ALLOWED_FILE_EXTENSIONS

        return (
            os.path.isfile(file_path) and os.path.splitext(file_path)[1] in extensions
        )

    @classmethod
    def parse_file_name(
        cls, file_path: str, require_sample_no: bool = True
    ) -> tuple[int, list[str]] | None:
        """Parses the file name to check if it contains a date (YYYYMMDD) and sample number (S##).

        Returns a tuple if ok:
        - int: Sample number
        - list[str]: Parts of the file name
        """
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
        else:
            return 1, name_parts

    @classmethod
    def files_from_folder(
        cls, folder_path: str, extensions: set[str] | None = None
    ) -> list[str]:
        """Returns a list of file paths from the specified folder, filtering by allowed extensions.

        Args:
            folder_path: Path to folder to search
            extensions: Set of allowed file extensions (default: cls._ALLOWED_FILE_EXTENSIONS)

        Returns:
            List of file paths matching the criteria

        Raises:
            FileNotFoundError: If folder doesn't exist
        """
        if extensions is None:
            extensions = cls._ALLOWED_FILE_EXTENSIONS

        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        files = []
        for item in os.listdir(folder_path):
            path = os.path.join(folder_path, item)
            if cls.is_allowed_file(path, extensions):
                files.append(path)

        return files
