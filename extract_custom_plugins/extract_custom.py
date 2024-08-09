import re
import json

def extract_custom_plugins(md_content):
    custom_plugins = {}
    lines = md_content.splitlines()
    current_plugin = None

    for i, line in enumerate(lines):

        # Check for H3 headers (for custom plugin names)
        h3_match = re.match(r'###\s+(.*)', line)
        if h3_match:
            header = h3_match.group(1).strip()
            if "custom" in header.lower():
                current_plugin = "Custom Plugins"  # This is a category, not a specific plugin
            else:
                current_plugin = None  # Reset if no relevant header
            continue

        # Check for H4 headers (for specific plugin names) only if inside custom plugins
        if current_plugin == "Custom Plugins":  # Ensure we are within the custom plugins context
            h4_match = re.match(r'####\s+(.*)', line)
            if h4_match:
                sub_header = h4_match.group(1).strip()

                # Set the current plugin to the specific plugin name
                current_plugin_name = sub_header
                custom_plugins[current_plugin_name] = {
                    'description': '',  # Initialize description as an empty string
                    'python_code': ''   # Initialize python_code as an empty string
                }  # Initialize the plugin entry

                # Move to the next line to capture description or code
                i += 1
                while i < len(lines) and not lines[i].startswith('####') and not re.match(r'###\s+(.*)', lines[i]):
                    # Check for **Description:** and **Python Code:** in the line
                    if '**description:**' in lines[i].lower():
                        i += 1  # Move to the next line to capture the actual description
                        description_lines = []
                        while i < len(lines) and not lines[i].startswith('**') and not re.match(r'###\s+(.*)', lines[i]):
                            description_lines.append(lines[i].strip())
                            i += 1
                        custom_plugins[current_plugin_name]['description'] = ' '.join(description_lines).strip()  # Add description to the dict
                    elif '**python code:**' in lines[i].lower():
                        i += 1  # Move to the next line to capture the actual code
                        if i < len(lines) and lines[i].strip() == '```python':
                            i += 1  # Skip the opening ```python line
                        code_lines = []
                        while i < len(lines) and not lines[i].strip() == '```':
                            code_lines.append(lines[i].strip())
                            i += 1
                        if i < len(lines) and lines[i].strip() == '```':
                            i += 1  # Skip the closing ``` line
                        custom_plugins[current_plugin_name]['python_code'] = '\n'.join(code_lines).strip()  # Add code to the dict
                    else:
                        i += 1  # Move to the next line if no match
                continue

    return {'Custom Plugins': custom_plugins}

# Function to read the Markdown file and process it
def process_markdown_file(file_path):
    with open(file_path, 'r') as file:
        md_content = file.read()
    return extract_custom_plugins(md_content)

# Main function
if __name__ == "__main__":
    file_path = 'arch_to_pipeline3.md'  # Adjust path if necessary
    result = process_markdown_file(file_path)
    print(json.dumps(result, indent=4))