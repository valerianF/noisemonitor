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
    
    # Add docstring
    docstring = func_info['docstring']
    
    # Parse docstring sections
    lines = docstring.split('\n')
    description = []
    in_params = False
    in_returns = False
    params = []
    returns = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('Parameters') or line == 'Parameters':
            in_params = True
            in_returns = False
            i += 1
            if i < len(lines) and lines[i].strip().startswith('---'):
                i += 1
            continue
        elif line.startswith('Returns') or line == 'Returns':
            in_params = False
            in_returns = True
            i += 1
            if i < len(lines) and lines[i].strip().startswith('---'):
                i += 1
            continue
        elif line.startswith('Note') or line.startswith('Example'):
            in_params = False
            in_returns = False
        
        if in_params:
            if line:
                params.append(line)
        elif in_returns:
            if line:
                returns.append(line)
        elif not in_params and not in_returns and line:
            description.append(line)
        
        i += 1
    
    # Add description
    if description:
        md += ' '.join(description) + "\n\n"
    
    # Add parameters
    if params:
        md += "**Parameters:**\n"
        current_param = []
        for line in params:
            if line and not line.startswith(' ') and ':' in line:
                if current_param:
                    # Extract parameter name and wrap in backticks
                    param_text = ' '.join(current_param)
                    if ':' in param_text:
                        param_name = param_text.split(':')[0].strip()
                        param_rest = ':'.join(param_text.split(':')[1:])
                        md += f"- `{param_name}`:{param_rest}\n"
                    else:
                        md += "- " + param_text + "\n"
                current_param = [line]
            else:
                current_param.append(line)
        if current_param:
            # Extract parameter name and wrap in backticks
            param_text = ' '.join(current_param)
            if ':' in param_text:
                param_name = param_text.split(':')[0].strip()
                param_rest = ':'.join(param_text.split(':')[1:])
                md += f"- `{param_name}`:{param_rest}\n"
            else:
                md += "- " + param_text + "\n"
        md += "\n"
    
    # Add returns
    if returns:
        md += "**Returns:**\n"
        md += "- " + ' '.join(returns) + "\n\n"
    
    return md


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
    
    print(f"✓ API documentation generated: {output_path}")
    print(f"  Total modules documented: {len(modules)}")


if __name__ == "__main__":
    generate_api_docs()
