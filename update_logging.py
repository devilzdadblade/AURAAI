"""
Script to update all files to use the centralized logging configuration.
This script will find and replace redundant logging configurations with a simple logger = logging.getLogger(__name__) line.
"""
import os
import re
import glob

def has_handler_config(lines):
    """Check if handler configuration exists in the lines."""
    for line in lines:
        if re.search(r'handler\s*=\s*logging\.StreamHandler\(\)', line):
            return True
    return False

def is_logger_get_logger(line):
    return re.search(r'logger\s*=\s*logging\.getLogger\(__name__\)', line)

def is_handler_stream_handler(line):
    return re.search(r'handler\s*=\s*logging\.StreamHandler\(\)', line)

def is_logger_add_handler(line):
    return re.search(r'logger\.addHandler\(handler\)', line)

def is_logger_set_level(line):
    return re.search(r'logger\.setLevel\(logging\.[A-Z]+\)', line)

def find_handler_block_end(lines, start_idx):
    """Find the end index of the handler block."""
    for j in range(start_idx, len(lines)):
        if is_logger_add_handler(lines[j]):
            return j
    return -1

def has_handler_block_ahead(lines, idx):
    """Check if handler block exists within the next 10 lines."""
    for j in range(idx + 1, min(idx + 11, len(lines))):
        if is_handler_stream_handler(lines[j]):
            return True
    return False

def remove_handler_block(lines):
    """Remove handler configuration block from lines."""
    new_content = []
    i = 0
    modified = False
    while i < len(lines):
        line = lines[i]
        if is_logger_get_logger(line):
            new_content.append(line)
            if has_handler_block_ahead(lines, i):
                end_idx = find_handler_block_end(lines, i + 1)
                if end_idx != -1:
                    i = end_idx  # Skip to this line
                    modified = True
        elif is_logger_set_level(line) and modified:
            pass  # Skip this line
        else:
            new_content.append(line)
        i += 1
    return new_content, modified

def write_if_modified(file_path, content, original_content):
    """Write content to file if it was modified."""
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated {file_path}")
        return True
    return False

def update_file(file_path):
    """Update a file to use the centralized logging configuration."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        original_content = content
        lines = content.split('\n')
        if has_handler_config(lines):
            print(f"Found handler configuration in {file_path}")
            new_content, modified = remove_handler_block(lines)
            if modified:
                content = '\n'.join(new_content)
        return write_if_modified(file_path, content, original_content)
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def main():
    """Main function to update all Python files."""
    # Find all Python files in the src directory
    python_files = glob.glob('src/**/*.py', recursive=True)
    
    # Update each file
    updated_count = 0
    for file_path in python_files:
        if update_file(file_path):
            updated_count += 1
    
    print(f"Updated {updated_count} out of {len(python_files)} files.")

if __name__ == '__main__':
    main()