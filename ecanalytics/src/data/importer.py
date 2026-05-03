import os


class Importer:
    __Allowed_File_Extensions = {".txt", ".csv"}

    @classmethod
    def Is_Allowed_File(cls, file_path: str, extensions: set[str] = None) -> bool:
        """
        Checks if the file has an allowed extension.
        """
        if extensions is None:
            extensions = cls.__Allowed_File_Extensions

        return (
            os.path.isfile(file_path) and os.path.splitext(file_path)[1] in extensions
        )

    @classmethod
    def Parse_File_Name(
        cls, file_path: str, require_sample_no: bool = True
    ) -> tuple[int, list[str]] | None:
        """
        Parses the file name to check if it contains a date (YYYYMMDD) and sample number (S##).

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
    def Files_From_Folder(
        cls, folder_path: str, extensions: set[str] | None = None
    ) -> list[str]:
        """Returns a list of file paths from the specified folder, filtering by allowed extensions.

        Args:
            folderPath: Path to folder to search
            extensions: Set of allowed file extensions (default: cls.AllowedFileExtensions)

        Returns:
            List of file paths matching the criteria

        Raises:
            FileNotFoundError: If folder doesn't exist
        """
        if extensions is None:
            extensions = cls.__Allowed_File_Extensions

        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        files = []
        for item in os.listdir(folder_path):
            path = os.path.join(folder_path, item)
            if cls.Is_Allowed_File(path, extensions):
                files.append(path)

        return files
