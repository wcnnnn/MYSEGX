from setuptools import setup, find_packages
import os

# 读取版本号
def get_version():
    init_py_path = os.path.join(os.path.dirname(__file__), 'MYSEGX', '__init__.py')
    with open(init_py_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[-1].strip().strip('"\'')
    return '0.1.0'

# 读取依赖
def get_requirements():
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='MYSEGX',
    version=get_version(),
    packages=find_packages(exclude=['tests*', 'docs*']),
    install_requires=get_requirements(),
    
    # 项目元数据
    author='chuanning wang',
    author_email='3076648528@qq.com',
    description='现代化图像分割框架，支持DETR、UNet和CNN等多种模型',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/wcnnnn/MYSEGX',
    project_urls={
        'Documentation': 'https://mysegx.readthedocs.io/',
        'Source': 'https://github.com/wcnnnn/MYSEGX',
        'Tracker': 'https://github.com/wcnnnn/MYSEGX/issues',
    },
    
    # 分类信息
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
    ],
    
    # 项目配置
    python_requires='>=3.7',
    include_package_data=True,  # 包含 MANIFEST.in 中指定的数据文件
    zip_safe=False,  # 不以 zip 格式安装
    
    # 命令行工具
    entry_points={
        'console_scripts': [
            'mysegx=MYSEGX.cli:main',  # 添加命令行接口
        ],
    },
    
    # 测试配置
    test_suite='tests',
    tests_require=[
        'pytest>=6.0',
        'pytest-cov>=2.0',
    ],
)