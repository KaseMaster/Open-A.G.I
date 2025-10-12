#!/usr/bin/env python3
"""
Script para debuggear el problema con los módulos de email
"""

import sys

def debug_email_modules():
    print(f'Python version: {sys.version}')
    print(f'Python executable: {sys.executable}')
    
    try:
        import email
        print(f'✅ Email module path: {email.__file__}')
        
        import email.mime
        print(f'✅ Email.mime module path: {email.mime.__file__}')
        
        # Intentar importar específicamente
        from email.mime import text
        print(f'✅ Email.mime.text module: {text}')
        
        # Verificar contenido del módulo
        import email.mime.text as emt
        print(f'MimeText available: {hasattr(emt, "MimeText")}')
        if hasattr(emt, 'MimeText'):
            print(f'✅ MimeText class: {emt.MimeText}')
        else:
            print(f'❌ Available attributes: {dir(emt)}')
            
        # Intentar importar directamente
        from email.mime.text import MimeText
        print(f'✅ Direct import successful: {MimeText}')
        
        # Probar crear una instancia
        msg = MimeText("Test message")
        print(f'✅ MimeText instance created: {type(msg)}')
        
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_email_modules()