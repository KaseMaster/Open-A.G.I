#!/usr/bin/env python3
"""
Script para reparar el archivo pyproject.toml con errores de sintaxis
"""

import re
import os

def fix_pyproject_toml():
    """Repara errores de sintaxis en pyproject.toml"""
    
    # Leer el archivo original
    with open('pyproject.toml', 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("🔧 Reparando pyproject.toml...")
    
    # Problemas comunes en configuración TOML
    fixes_applied = []
    
    # 1. Corregir configuración de pre-commit mal formateada
    if '[tool.pre-commit]' in content:
        # Buscar y corregir la sección problemática
        pattern = r'(\[tool\.pre-commit\]\s*repos\s*=\s*\[\s*\{[^}]*\}[^]]*\])'
        
        # Reemplazar con configuración válida
        fixed_precommit = '''[tool.pre-commit]
repos = [
    {repo = "https://github.com/pre-commit/pre-commit-hooks", rev = "v4.4.0", hooks = [
        {id = "trailing-whitespace"},
        {id = "end-of-file-fixer"},
        {id = "check-yaml"},
        {id = "check-added-large-files"},
        {id = "check-json"},
        {id = "check-toml"},
        {id = "check-xml"},
        {id = "mixed-line-ending"},
        {id = "check-case-conflict"},
        {id = "check-merge-conflict"},
        {id = "debug-statements"},
        {id = "requirements-txt-fixer"}
    ]},
    {repo = "https://github.com/psf/black", rev = "23.12.1", hooks = [
        {id = "black"}
    ]},
    {repo = "https://github.com/pycqa/isort", rev = "5.13.2", hooks = [
        {id = "isort"}
    ]},
    {repo = "https://github.com/pycqa/flake8", rev = "7.0.0", hooks = [
        {id = "flake8"}
    ]}
]'''
        
        # Encontrar el inicio y final de la sección problemática
        start_idx = content.find('[tool.pre-commit]')
        if start_idx != -1:
            # Buscar el final de la sección (próxima sección o final del archivo)
            next_section = content.find('\n[', start_idx + 1)
            if next_section == -1:
                next_section = len(content)
            
            # Reemplazar la sección completa
            content = content[:start_idx] + fixed_precommit + content[next_section:]
            fixes_applied.append("Configuración pre-commit corregida")
    
    # 2. Limpiar caracteres problemáticos
    # Remover caracteres de control no imprimibles
    content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', content)
    fixes_applied.append("Caracteres de control removidos")
    
    # 3. Normalizar espacios y líneas
    content = re.sub(r'\r\n', '\n', content)  # Normalizar line endings
    content = re.sub(r'\n{3,}', '\n\n', content)  # Reducir líneas vacías múltiples
    fixes_applied.append("Espacios y líneas normalizados")
    
    # 4. Validar sintaxis básica de llaves
    brace_count = content.count('{') - content.count('}')
    bracket_count = content.count('[') - content.count(']')
    
    if brace_count != 0:
        print(f"⚠️ Advertencia: Desbalance de llaves: {brace_count}")
    if bracket_count != 0:
        print(f"⚠️ Advertencia: Desbalance de corchetes: {bracket_count}")
    
    # Escribir el archivo corregido
    with open('pyproject.toml', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Correcciones aplicadas:")
    for fix in fixes_applied:
        print(f"   • {fix}")
    
    return True

def validate_toml():
    """Valida la sintaxis del archivo TOML corregido"""
    try:
        import tomllib
        with open('pyproject.toml', 'rb') as f:
            tomllib.load(f)
        print("✅ pyproject.toml sintaxis válida")
        return True
    except Exception as e:
        print(f"❌ Error de sintaxis: {e}")
        return False

def main():
    print("🛠️ AEGIS pyproject.toml Repair Tool")
    print("=" * 40)
    
    if not os.path.exists('pyproject.toml'):
        print("❌ pyproject.toml no encontrado")
        return False
    
    # Crear respaldo
    if not os.path.exists('pyproject.toml.backup'):
        import shutil
        shutil.copy2('pyproject.toml', 'pyproject.toml.backup')
        print("📋 Respaldo creado: pyproject.toml.backup")
    
    # Aplicar correcciones
    if fix_pyproject_toml():
        # Validar resultado
        if validate_toml():
            print("\n🎉 pyproject.toml reparado exitosamente!")
            return True
        else:
            print("\n❌ La reparación falló. Restaurando respaldo...")
            import shutil
            shutil.copy2('pyproject.toml.backup', 'pyproject.toml')
            return False
    
    return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)