def txt_to_list(file_path):
    """
    Reads a file and returns its contents as a list of strings.
    
    Args:
        file_path (str): The path to the file to be read.
        
    Returns:
        list: A list of strings, where each string represents a line from the file.
    """
    file_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            file_list.append(line.strip())
    return file_list