import os
import yaml

def load_prompt_from_yaml(prompt_name):
    file_path = os.path.join(os.path.dirname(__file__), "../prompts", "prompts.yaml")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            yaml_content = yaml.safe_load(file)
            return yaml_content.get(prompt_name, '')
    except Exception as e:
        print(f"Error loading prompt from YAML: {e}")
        return ""