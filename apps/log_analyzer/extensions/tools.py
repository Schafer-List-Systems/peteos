
from pathlib import Path


def read_file(filename: str) -> str:
    """
    This tool allows you to read the content of a file.
    
    Args:
        filename (str): The path to the file to read.
    
    Returns:
        str: The content of the file.
    """
    try:
        with open(filename, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

def update_file_content(filename: str, old_content: str, new_content: str) -> str:
    """
    This tool allows you to update the content of a file by replacing old content with new content.
    
    Args:
        filename (str): The path to the file to update.
        old_content (str): The string to be replaced.
        new_content (str): The string to replace with.
    
    Returns:
        str: A confirmation message indicating the result.
    """
    if not Path(filename).exists():
        return f"File {filename} does not exist."
    
    try:
        with open(filename, 'r') as f:
            content = f.read()
            
        if old_content in content:
            new_file_content = content.replace(old_content, new_content)
            with open(filename, 'w') as f:
                f.write(new_file_content)
            return f"Successfully updated {filename}."
        else:
            return f"Could not find '{old_content}' in {filename}."
    except Exception as e:
        return f"Error updating file: {e}"
