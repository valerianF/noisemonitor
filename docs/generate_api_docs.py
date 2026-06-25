"""
Automated API documentation generator for noisemonitor package.
Extracts docstrings from source code and generates markdown documentation.
"""

import os
import ast
import inspect
from pathlib import Path


def extract_docstring_info(node):
    """Extract function signature and docstring from AST node."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    
    # Get function name
    func_name = node.name
    
    # Skip private functions
    if func_name.startswith('_') and not func_name.startswith('__'):
        return None
    
    # Get docstring
    docstring = ast.get_docstring(node)
    if not docstring:
        return None
    
    # Get function signature
    args = []
    defaults_offset = len(node.args.args) - len(node.args.defaults)
    
    for i, arg in enumerate(node.args.args):
        arg_name = arg.arg
        if arg_name == 'self':
            continue
            
        # Check if there's a default value
        default_idx = i - defaults_offset
        if default_idx >= 0 and default_idx < len(node.args.defaults):
            default = node.args.defaults[default_idx]
            try:
                default_val = ast.literal_eval(default)
                args.append(f"{arg_name}={repr(default_val)}")
            except:
                args.append(f"{arg_name}=...")
        else:
            args.append(arg_name)
    
    signature = f"{func_name}({', '.join(args)})"
    
    return {
        'name': func_name,
        'signature': signature,
        'docstring': docstring,
        'is_async': isinstance(node, ast.AsyncFunctionDef)
    }


def parse_module(filepath):
    """Parse a Python module and extract all public functions."""
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
    
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    
    functions = []
    for node in ast.walk(tree):
        func_info = extract_docstring_info(node)
        if func_info:
            functions.append(func_info)
    
    return functions


def format_function_doc(module_path, func_info):
    """Format a function's documentation as markdown."""
    async_prefix = "async " if func_info['is_async'] else ""
    md = f"### `{module_path}.{func_info['name']}()`\n\n"
    
    # Parse docstring sections
    docstring = func_info['docstring']
    lines = docstring.split('\n')
    
    # Find section boundaries
    params_idx = None
    returns_idx = None
    notes_idx = None
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == 'Parameters':
            params_idx = i
        elif stripped == 'Returns':
            returns_idx = i
        elif stripped == 'Notes':
            notes_idx = i
    
    # Extract description (everything before Parameters)
    description_end = params_idx if params_idx else len(lines)
    description_lines = lines[:description_end]
    
    # Remove trailing empty lines from description
    while description_lines and not description_lines[-1].strip():
        description_lines.pop()
    
    # Join description, removing leading/trailing whitespace
    description = '\n'.join(description_lines).strip()
    if description:
        md += description + "\n\n"
    
    # Extract and format parameters
    if params_idx:
        params_end = returns_idx if returns_idx else (notes_idx if notes_idx else len(lines))
        params_section = lines[params_idx + 2:params_end]  # Skip "Parameters" and "---"
        
        if params_section:
            md += "**Parameters:**\n"
            current_param = None
            
            for line in params_section:
                stripped = line.strip()
                
                # Skip empty lines and dashes
                if not stripped or stripped.startswith('---'):
                    continue
                
                # Check if this is a new parameter (doesn't start with whitespace in original)
                if line and line[0] not in (' ', '\t'):
                    # Save previous parameter if exists
                    if current_param:
                        md += format_parameter(current_param) + "\n"
                    current_param = [line]
                elif current_param:
                    # Continuation of parameter description
                    current_param.append(line)
            
            # Add last parameter
            if current_param:
                md += format_parameter(current_param) + "\n"
            
            md += "\n"
    
    # Extract and format returns
    if returns_idx:
        returns_end = notes_idx if notes_idx else len(lines)
        returns_section = lines[returns_idx + 2:returns_end]  # Skip "Returns" and "---"
        
        returns_text = '\n'.join(returns_section).strip()
        if returns_text:
            md += "**Returns:**\n"
            md += "- " + returns_text + "\n\n"
    
    return md


def format_parameter(param_lines):
    """Format a single parameter from NumPy docstring format."""
    # Join all lines
    full_text = '\n'.join(param_lines)
    
    # Split on first colon to separate name/type from description
    if ':' in full_text:
        colon_idx = full_text.index(':')
        name_type = full_text[:colon_idx].strip()
        description = full_text[colon_idx + 1:].strip()
        
        # Extract just the parameter name (before the type)
        param_name = name_type.split()[0] if name_type else "unknown"
        
        # Format with proper indentation for multi-line descriptions
        desc_lines = description.split('\n')
        formatted_desc = desc_lines[0]
        
        if len(desc_lines) > 1:
            # Add indentation for continuation lines
            for line in desc_lines[1:]:
                stripped = line.strip()
                if stripped:
                    formatted_desc += ' \\\n  ' + stripped
        
        return f"- `{param_name}`: {name_type.split(maxsplit=1)[1] if ' ' in name_type else ''} {formatted_desc}".rstrip()
    else:
        return f"- {full_text.strip()}"



def generate_api_docs():
    """Generate complete API documentation."""
    src_path = Path(__file__).parent.parent / "src" / "noisemonitor"
    
    # Define modules to document
    modules = {
        "Loading Module": {
            "path": "noisemonitor",
            "files": ["util/load.py"]
        },
        "Filter Module": {
            "path": "noisemonitor.filter",
            "files": ["util/filter.py"]
        },
        "Summary Module": {
            "path": "noisemonitor.summary",
            "files": ["summary.py"]
        },
        "Profile Module": {
            "path": "noisemonitor.profile",
            "files": ["profile.py"]
        },
        "Display Module": {
            "path": "noisemonitor.display",
            "files": ["util/display.py"]
        },
        "Core Module": {
            "path": "noisemonitor.util.core",
            "files": ["util/core.py"]
        },
        "Weather Module": {
            "path": "noisemonitor.weather.weathercan",
            "files": ["weather/weathercan.py"]
        }
    }
    
    # Start markdown document
    md_content = "# API Reference\n\n"
    md_content += "Complete reference for all noisemonitor functions and modules.\n\n"
    md_content += "**Note:** This documentation is auto-generated from source code docstrings.\n\n"
    
    # Table of contents
    md_content += "## Table of Contents\n"
    for module_name in modules.keys():
        anchor = module_name.lower().replace(" ", "-")
        md_content += f"- [{module_name}](#{anchor})\n"
    md_content += "\n"
    
    # Generate documentation for each module
    for module_name, module_info in modules.items():
        md_content += f"## {module_name}\n\n"
        
        for file_rel_path in module_info["files"]:
            file_path = src_path / file_rel_path
            if not file_path.exists():
                continue
            
            functions = parse_module(file_path)
            
            # Sort functions alphabetically
            functions.sort(key=lambda x: x['name'])
            
            for func_info in functions:
                func_doc = format_function_doc(module_info["path"], func_info)
                md_content += func_doc
    
    # Write to file
    output_path = Path(__file__).parent / "api.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"[OK] API documentation generated: {output_path}")
    print(f"  Total modules documented: {len(modules)}")


if __name__ == "__main__":
    generate_api_docs()
