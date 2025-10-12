#!/usr/bin/env python3
"""
Script para verificar el estado de la base de datos de alertas
"""

import sqlite3
import os
from datetime import datetime

def check_alerts_database():
    print('🔍 Verificando base de datos de alertas...')
    
    if os.path.exists('alerts.db'):
        conn = sqlite3.connect('alerts.db')
        cursor = conn.cursor()
        
        # Verificar si la tabla existe
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='alerts'")
        table_exists = cursor.fetchone()
        
        if table_exists:
            cursor.execute('SELECT COUNT(*) FROM alerts')
            count = cursor.fetchone()[0]
            print(f'📊 Total de alertas en la base de datos: {count}')
            
            if count > 0:
                cursor.execute('SELECT id, title, severity, category, timestamp FROM alerts ORDER BY timestamp DESC LIMIT 10')
                alerts = cursor.fetchall()
                print('\n🚨 ÚLTIMAS ALERTAS:')
                for alert in alerts:
                    timestamp = datetime.fromtimestamp(float(alert[4])).strftime('%Y-%m-%d %H:%M:%S')
                    print(f'  - ID: {alert[0]}')
                    print(f'    Título: {alert[1]}')
                    print(f'    Severidad: {alert[2]}')
                    print(f'    Categoría: {alert[3]}')
                    print(f'    Timestamp: {timestamp}')
                    print()
            else:
                print('ℹ️ No hay alertas en la base de datos')
                
            # Mostrar esquema de la tabla
            cursor.execute("PRAGMA table_info(alerts)")
            columns = cursor.fetchall()
            print('\n📋 Esquema de la tabla alerts:')
            for col in columns:
                print(f'  - {col[1]} ({col[2]})')
                
        else:
            print('❌ La tabla de alertas no existe')
            
            # Mostrar todas las tablas disponibles
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            if tables:
                print('\n📋 Tablas disponibles:')
                for table in tables:
                    print(f'  - {table[0]}')
            else:
                print('ℹ️ No hay tablas en la base de datos')
        
        conn.close()
    else:
        print('❌ La base de datos de alertas no existe')

if __name__ == "__main__":
    check_alerts_database()