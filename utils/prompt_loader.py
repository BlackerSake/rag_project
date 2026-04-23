import os
import yaml
import time
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class PromptLoader:
    """Prompt加载器，支持从YAML文件加载和管理prompt模板"""
    
    def __init__(self, yaml_path=None):
        """
        初始化PromptLoader
        
        Args:
            yaml_path: prompt配置文件路径
        """
        # 使用绝对路径，确保在任何工作目录下都能找到配置文件
        if yaml_path is None:
            # 获取当前文件所在目录的父目录，然后拼接config/prompt.yaml
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            self.yaml_path = os.path.join(project_root, "config", "prompt.yaml")
        else:
            self.yaml_path = yaml_path
        
        self.prompts = {}
        self.last_modified = 0
        self.load_prompts()
    
    def load_prompts(self):
        """加载prompt配置"""
        try:
            print(f"尝试加载prompt配置文件: {self.yaml_path}")
            if not os.path.exists(self.yaml_path):
                print(f"配置文件不存在: {self.yaml_path}")
                return
            
            with open(self.yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            if not data or 'prompts' not in data:
                print("配置文件格式错误，缺少prompts字段")
                return
            
            self.prompts = {}
            for prompt_name, prompt_config in data.get('prompts', {}).items():
                messages = []
                for role, content in prompt_config.items():
                    if role == 'system':
                        # 处理多行字符串
                        if isinstance(content, list):
                            content = ''.join(content)
                        messages.append(("system", content))
                    elif role == 'user':
                        # 处理多行字符串
                        if isinstance(content, list):
                            content = ''.join(content)
                        messages.append(("user", content))
                    elif role == 'messages':
                        messages.append(MessagesPlaceholder(variable_name=content))
                
                self.prompts[prompt_name] = ChatPromptTemplate.from_messages(messages)
            
            # 更新最后修改时间
            self.last_modified = os.path.getmtime(self.yaml_path)
            print(f"成功加载{len(self.prompts)}个prompt模板")
            print(f"加载的prompt模板: {list(self.prompts.keys())}")
        except Exception as e:
            print(f"加载prompt配置失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def get_prompt(self, prompt_name):
        """
        获取指定名称的prompt模板
        
        Args:
            prompt_name: prompt名称
            
        Returns:
            ChatPromptTemplate对象
        """
        # 检查文件是否被修改，如果是则重新加载
        if os.path.getmtime(self.yaml_path) > self.last_modified:
            self.load_prompts()
        
        return self.prompts.get(prompt_name)
    
    def get_all_prompts(self):
        """
        获取所有prompt模板
        
        Returns:
            dict: 所有prompt模板
        """
        # 检查文件是否被修改，如果是则重新加载
        if os.path.getmtime(self.yaml_path) > self.last_modified:
            self.load_prompts()
        
        return self.prompts

# 全局PromptLoader实例
prompt_loader = PromptLoader()
