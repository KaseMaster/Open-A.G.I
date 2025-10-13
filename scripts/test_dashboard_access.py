#!/usr/bin/env python3
"""
Script para probar el acceso al dashboard AEGIS con autenticación
"""

import requests
import json
import time
from requests.auth import HTTPBasicAuth

def test_dashboard_access():
    """Prueba el acceso al dashboard con diferentes métodos de autenticación"""
    
    dashboard_url = "http://127.0.0.1:8090"
    username = "admin"
    password = "aegis123"
    
    print("🔍 Probando acceso al Dashboard AEGIS...")
    print(f"📊 URL: {dashboard_url}")
    print(f"👤 Usuario: {username}")
    
    # Test 1: Acceso sin autenticación
    print("\n1️⃣ Probando acceso sin autenticación...")
    try:
        response = requests.get(dashboard_url, timeout=5)
        print(f"   Status: {response.status_code}")
        if response.status_code == 401:
            print("   ✅ Autenticación requerida (esperado)")
        else:
            print(f"   ⚠️ Respuesta inesperada: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Error de conexión: {e}")
        return False
    
    # Test 2: Acceso con Basic Auth
    print("\n2️⃣ Probando acceso con Basic Auth...")
    try:
        response = requests.get(
            dashboard_url, 
            auth=HTTPBasicAuth(username, password),
            timeout=5
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ Acceso exitoso con Basic Auth")
            print(f"   📄 Content-Type: {response.headers.get('Content-Type', 'N/A')}")
            return True
        else:
            print(f"   ❌ Fallo en autenticación: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Error de conexión: {e}")
    
    # Test 3: Login via POST
    print("\n3️⃣ Probando login via POST...")
    try:
        session = requests.Session()
        login_data = {
            'username': username,
            'password': password
        }
        
        response = session.post(
            f"{dashboard_url}/login",
            data=login_data,
            timeout=5
        )
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200 or response.status_code == 302:
            print("   ✅ Login exitoso")
            
            # Probar acceso a la página principal con sesión
            response = session.get(dashboard_url, timeout=5)
            print(f"   Dashboard access: {response.status_code}")
            if response.status_code == 200:
                print("   ✅ Acceso al dashboard con sesión exitoso")
                return True
        else:
            print(f"   ❌ Fallo en login: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Error de conexión: {e}")
    
    # Test 4: Verificar endpoints específicos
    print("\n4️⃣ Probando endpoints específicos...")
    endpoints = ['/api/status', '/api/metrics', '/api/system']
    
    for endpoint in endpoints:
        try:
            response = requests.get(
                f"{dashboard_url}{endpoint}",
                auth=HTTPBasicAuth(username, password),
                timeout=5
            )
            print(f"   {endpoint}: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"   {endpoint}: Error - {e}")
    
    return False

def test_secure_chat_access():
    """Prueba el acceso al chat seguro"""
    
    chat_url = "http://[::1]:5173"
    
    print("\n💬 Probando acceso al Chat Seguro...")
    print(f"📊 URL: {chat_url}")
    
    try:
        response = requests.get(chat_url, timeout=5)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ Chat seguro accesible")
            print(f"   📄 Content-Type: {response.headers.get('Content-Type', 'N/A')}")
            return True
        else:
            print(f"   ⚠️ Respuesta: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Error de conexión: {e}")
    
    return False

def main():
    """Función principal"""
    print("=" * 60)
    print("🔧 AEGIS - Test de Acceso a Servicios Web")
    print("=" * 60)
    
    dashboard_ok = test_dashboard_access()
    chat_ok = test_secure_chat_access()
    
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE RESULTADOS")
    print("=" * 60)
    print(f"📊 Dashboard: {'✅ FUNCIONANDO' if dashboard_ok else '❌ PROBLEMA'}")
    print(f"💬 Chat Seguro: {'✅ FUNCIONANDO' if chat_ok else '❌ PROBLEMA'}")
    
    if dashboard_ok and chat_ok:
        print("\n🎉 Todos los servicios web están funcionando correctamente!")
    elif dashboard_ok or chat_ok:
        print("\n⚠️ Algunos servicios tienen problemas de conectividad.")
    else:
        print("\n❌ Ambos servicios tienen problemas de conectividad.")
    
    print("\n💡 RECOMENDACIONES:")
    if not dashboard_ok:
        print("   • Verificar que el dashboard esté ejecutándose en puerto 8000")
        print("   • Confirmar credenciales: admin/aegis123")
        print("   • Revisar logs del dashboard para errores")
    
    if not chat_ok:
        print("   • Verificar que el chat UI esté ejecutándose en puerto 3000")
        print("   • Confirmar que npm run dev esté activo")
        print("   • Revisar logs del proceso Node.js")

if __name__ == "__main__":
    main()