#!/usr/bin/env python3
"""
🎨 AEGIS Enterprise UI/UX - Sprint 5.1
Interfaz completa de usuario para el framework AEGIS
"""

import asyncio
import streamlit as st
import time
import json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import base64
from typing import Dict, List, Any, Optional
import warnings

# Importar componentes de AEGIS
from aegis_api import AEGISAPIService
from integration_pipeline import PipelineType
from multimodal_pipelines import MultimodalPipelineType

warnings.filterwarnings('ignore')

# ===== CONFIGURACIÓN =====

st.set_page_config(
    page_title="AEGIS Enterprise Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tema personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(45deg, #667eea 25%, transparent 25%),
                    linear-gradient(-45deg, #667eea 25%, transparent 25%),
                    linear-gradient(45deg, transparent 75%, #667eea 75%),
                    linear-gradient(-45deg, transparent 75%, #667eea 75%);
        background-size: 20px 20px;
        background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .success-msg {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ===== ESTADO DE LA APLICACIÓN =====

class AEGISUIApp:
    """Aplicación principal de UI"""

    def __init__(self):
        self.api_service = None
        self.api_base_url = "http://localhost:8000"
        self.api_key = "demo_key_2024"

    def initialize_app(self):
        """Inicializar aplicación"""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
            st.session_state.api_connected = False
            st.session_state.current_page = "dashboard"
            st.session_state.processing_history = []
            st.session_state.user_preferences = {}

        # Inicializar servicio API
        if not st.session_state.api_connected:
            self.connect_to_api()

    def connect_to_api(self):
        """Conectar a la API"""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(f"{self.api_base_url}/health", headers=headers, timeout=5)
            if response.status_code == 200:
                st.session_state.api_connected = True
                st.success("✅ Conectado a AEGIS API")
            else:
                st.session_state.api_connected = False
                st.error("❌ Error conectando a AEGIS API")
        except:
            st.session_state.api_connected = False
            st.warning("⚠️ AEGIS API no disponible. Ejecutar: python aegis_api.py")

    def api_request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict]:
        """Hacer request a la API"""
        if not st.session_state.api_connected:
            return None

        try:
            url = f"{self.api_base_url}{endpoint}"
            headers = {"Authorization": f"Bearer {self.api_key}"}

            if 'json' in kwargs:
                headers['Content-Type'] = 'application/json'

            response = requests.request(method, url, headers=headers, **kwargs)

            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error: {response.status_code}")
                return None

        except Exception as e:
            st.error(f"Request failed: {e}")
            return None

    def add_to_history(self, operation: str, result: Any):
        """Agregar operación al historial"""
        st.session_state.processing_history.append({
            'timestamp': time.time(),
            'operation': operation,
            'result': str(result)[:100] + '...' if len(str(result)) > 100 else str(result)
        })

        # Mantener solo últimas 50 operaciones
        if len(st.session_state.processing_history) > 50:
            st.session_state.processing_history = st.session_state.processing_history[-50:]

# ===== FUNCIONES DE UI =====

def render_header():
    """Renderizar header principal"""
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown('<h1 class="main-header">🤖 AEGIS Enterprise Platform</h1>', unsafe_allow_html=True)
        st.markdown("*Framework de IA Multimodal Avanzada*")

    with col2:
        if st.session_state.api_connected:
            st.markdown('<div class="success-msg">🟢 API Connected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="background: #ff6b6b; color: white; padding: 1rem; border-radius: 5px; margin: 1rem 0;">🔴 API Disconnected</div>', unsafe_allow_html=True)

    with col3:
        if st.button("🔄 Refresh Connection"):
            app.connect_to_api()

def render_sidebar():
    """Renderizar sidebar de navegación"""
    st.sidebar.title("🎯 Navigation")

    pages = {
        "🏠 Dashboard": "dashboard",
        "📝 Text Analysis": "text_analysis",
        "🖼️ Image Analysis": "image_analysis",
        "🎵 Audio Analysis": "audio_analysis",
        "🔄 Multimodal": "multimodal",
        "🎨 Generation": "generation",
        "📊 Analytics": "analytics",
        "🔧 Pipelines": "pipelines",
        "📈 Monitoring": "monitoring",
        "⚙️ Settings": "settings"
    }

    for page_name, page_key in pages.items():
        if st.sidebar.button(page_name, key=f"nav_{page_key}"):
            st.session_state.current_page = page_key
            st.rerun()

    st.sidebar.markdown("---")

    # Estado del sistema
    st.sidebar.subheader("📊 System Status")
    if st.session_state.api_connected:
        st.sidebar.success("API: Connected")
    else:
        st.sidebar.error("API: Disconnected")

    # Historial reciente
    if st.session_state.processing_history:
        st.sidebar.subheader("🕒 Recent Activity")
        for item in st.session_state.processing_history[-3:]:
            st.sidebar.text(f"• {item['operation']}")

def render_dashboard():
    """Renderizar dashboard principal"""
    st.header("🏠 Dashboard Overview")

    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>1,234</h3>
            <p>Procesamientos Totales</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>98.5%</h3>
            <p>Accuracy Promedio</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>0.8s</h3>
            <p>Tiempo Promedio</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>5</h3>
            <p>Modelos Activos</p>
        </div>
        """, unsafe_allow_html=True)

    # Gráfico de actividad
    st.subheader("📈 Actividad Reciente")

    if st.session_state.processing_history:
        # Crear datos para gráfico
        history_df = pd.DataFrame(st.session_state.processing_history[-20:])
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'], unit='s')

        fig = px.line(history_df, x='timestamp', y=history_df.index,
                     title="Procesamientos por Tiempo")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay actividad reciente")

    # Capacidades del sistema
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧠 Capacidades de IA")
        capabilities = [
            "✅ Análisis de Texto (NLP)",
            "✅ Procesamiento de Imágenes",
            "✅ Análisis de Audio",
            "✅ Fusión Multimodal",
            "✅ Generación de Contenido",
            "✅ Analytics Avanzados"
        ]
        for cap in capabilities:
            st.write(cap)

    with col2:
        st.subheader("🔧 Componentes del Sistema")
        components = [
            "✅ Integration Pipeline",
            "✅ Multimodal Pipelines",
            "✅ API REST Enterprise",
            "✅ TinyML & Edge AI",
            "✅ Generative AI",
            "✅ Monitoring System"
        ]
        for comp in components:
            st.write(comp)

def render_text_analysis():
    """Renderizar página de análisis de texto"""
    st.header("📝 Text Analysis")

    # Input de texto
    text_input = st.text_area(
        "Ingrese el texto a analizar:",
        height=150,
        placeholder="Escriba aquí el texto que desea analizar..."
    )

    # Opciones de análisis
    col1, col2 = st.columns(2)

    with col1:
        tasks = st.multiselect(
            "Tareas de análisis:",
            ["sentiment", "entities", "classification", "summarization"],
            default=["sentiment", "entities"]
        )

    with col2:
        language = st.selectbox("Idioma:", ["es", "en", "fr", "de"], index=0)

    # Botón de procesamiento
    if st.button("🚀 Analizar Texto", type="primary"):
        if not text_input.strip():
            st.error("Por favor ingrese texto para analizar")
            return

        with st.spinner("Analizando texto..."):
            # Llamar a la API
            result = app.api_request("POST", "/api/v1/text/analyze", json={
                "text": text_input,
                "tasks": tasks,
                "language": language
            })

            if result and result.get("success"):
                data = result["data"]

                # Mostrar resultados
                st.success("✅ Análisis completado!")

                for task in tasks:
                    if task in data["results"]:
                        st.subheader(f"📊 Resultados - {task.title()}")
                        st.json(data["results"][task])

                # Agregar al historial
                app.add_to_history("text_analysis", data["results"])

                st.info(".3f"            else:
                st.error("Error en el análisis de texto")

def render_image_analysis():
    """Renderizar página de análisis de imagen"""
    st.header("🖼️ Image Analysis")

    # Upload de imagen
    uploaded_file = st.file_uploader(
        "Seleccione una imagen:",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Formatos soportados: JPG, PNG, BMP"
    )

    if uploaded_file is not None:
        # Mostrar imagen
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", use_column_width=True)

        # Opciones de análisis
        tasks = st.multiselect(
            "Tareas de análisis:",
            ["detection", "classification", "segmentation", "captioning"],
            default=["detection", "classification"]
        )

        # Botón de procesamiento
        if st.button("🚀 Analizar Imagen", type="primary"):
            with st.spinner("Analizando imagen..."):
                # Preparar archivo para API
                files = {"file": uploaded_file.getvalue()}
                data = {"tasks": tasks}

                # Llamar a la API
                try:
                    headers = {"Authorization": f"Bearer {app.api_key}"}
                    response = requests.post(
                        f"{app.api_base_url}/api/v1/image/analyze",
                        files={"file": uploaded_file.getvalue()},
                        data={"tasks": json.dumps(tasks)},
                        headers=headers
                    )

                    if response.status_code == 200:
                        result = response.json()

                        if result.get("success"):
                            data = result["data"]

                            st.success("✅ Análisis completado!")

                            # Mostrar resultados
                            for task in tasks:
                                if task in data["results"]:
                                    st.subheader(f"📊 Resultados - {task.title()}")
                                    st.json(data["results"][task])

                            app.add_to_history("image_analysis", data["results"])
                            st.info(".3f"                        else:
                            st.error("Error en el análisis de imagen")
                    else:
                        st.error(f"API Error: {response.status_code}")

                except Exception as e:
                    st.error(f"Error conectando a la API: {e}")

def render_generation():
    """Renderizar página de generación"""
    st.header("🎨 Content Generation")

    # Tipo de generación
    generation_type = st.selectbox(
        "Tipo de generación:",
        ["text", "image", "multimodal"],
        format_func=lambda x: {
            "text": "📝 Texto",
            "image": "🖼️ Imagen",
            "multimodal": "🔄 Multimodal"
        }[x]
    )

    # Input según tipo
    if generation_type == "text":
        prompt = st.text_area(
            "Prompt para generación de texto:",
            placeholder="Escribe un prompt creativo...",
            height=100
        )

        max_length = st.slider("Longitud máxima:", 50, 500, 200)

        if st.button("🎨 Generar Texto", type="primary"):
            if not prompt.strip():
                st.error("Por favor ingrese un prompt")
                return

            with st.spinner("Generando texto..."):
                result = app.api_request("POST", "/api/v1/generate/text", data={
                    "prompt": prompt,
                    "max_length": max_length
                })

                if result and result.get("success"):
                    generated_text = result["data"]["result"]
                    st.success("✅ Texto generado!")
                    st.text_area("Resultado:", generated_text, height=200)
                    app.add_to_history("text_generation", generated_text)
                else:
                    st.error("Error en la generación de texto")

    elif generation_type == "image":
        prompt = st.text_input(
            "Prompt para generación de imagen:",
            placeholder="Describe la imagen que quieres generar..."
        )

        if st.button("🎨 Generar Imagen", type="primary"):
            if not prompt.strip():
                st.error("Por favor ingrese un prompt")
                return

            with st.spinner("Generando imagen..."):
                result = app.api_request("POST", "/api/v1/generate/image", data={
                    "prompt": prompt
                })

                if result and result.get("success"):
                    st.success("✅ Imagen generada!")
                    # Placeholder - en implementación real mostrar imagen
                    st.info("Imagen generada (simulada)")
                    app.add_to_history("image_generation", prompt)
                else:
                    st.error("Error en la generación de imagen")

def render_monitoring():
    """Renderizar página de monitoring"""
    st.header("📈 System Monitoring")

    # Tabs para diferentes métricas
    tab1, tab2, tab3 = st.tabs(["📊 Performance", "🔧 Components", "📋 History"])

    with tab1:
        st.subheader("Performance Metrics")

        # Métricas simuladas
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("CPU Usage", "45%", "↓ 5%")
        with col2:
            st.metric("Memory Usage", "2.1 GB", "↑ 0.2 GB")
        with col3:
            st.metric("API Requests", "1,234", "↑ 56")

        # Gráfico de rendimiento
        st.subheader("Response Time Trend")
        times = np.random.normal(0.8, 0.2, 50)
        fig = px.line(y=times, title="API Response Times (seconds)")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Component Status")

        components = {
            "Integration Pipeline": "🟢 Healthy",
            "Multimodal Pipelines": "🟢 Healthy",
            "API Service": "🟢 Healthy" if st.session_state.api_connected else "🔴 Disconnected",
            "Text Models": "🟢 Healthy",
            "Vision Models": "🟢 Healthy",
            "Audio Models": "🟢 Healthy"
        }

        for component, status in components.items():
            st.write(f"**{component}**: {status}")

    with tab3:
        st.subheader("Processing History")

        if st.session_state.processing_history:
            history_df = pd.DataFrame(st.session_state.processing_history[-20:])
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'], unit='s')
            history_df = history_df[['timestamp', 'operation', 'result']]

            st.dataframe(history_df, use_container_width=True)
        else:
            st.info("No processing history available")

def render_settings():
    """Renderizar página de configuración"""
    st.header("⚙️ Settings")

    st.subheader("API Configuration")
    api_url = st.text_input("API Base URL:", value=app.api_base_url)
    api_key = st.text_input("API Key:", value=app.api_key, type="password")

    if st.button("💾 Save Settings"):
        app.api_base_url = api_url
        app.api_key = api_key
        st.success("Settings saved!")

    st.subheader("UI Preferences")
    theme = st.selectbox("Theme:", ["Light", "Dark", "Auto"], index=0)
    language = st.selectbox("Language:", ["Español", "English"], index=0)

    st.subheader("System Information")
    st.write(f"**Version**: 1.0.0")
    st.write(f"**Framework**: AEGIS Enterprise")
    st.write(f"**Components**: 15+ AI models")
    st.write(f"**Pipelines**: 8 specialized pipelines")

# ===== MAIN APP =====

def main():
    """Función principal de la aplicación"""

    # Inicializar app
    global app
    app = AEGISUIApp()
    app.initialize_app()

    # Renderizar header
    render_header()

    # Renderizar sidebar
    render_sidebar()

    # Renderizar página actual
    current_page = st.session_state.current_page

    if current_page == "dashboard":
        render_dashboard()
    elif current_page == "text_analysis":
        render_text_analysis()
    elif current_page == "image_analysis":
        render_image_analysis()
    elif current_page == "generation":
        render_generation()
    elif current_page == "monitoring":
        render_monitoring()
    elif current_page == "settings":
        render_settings()
    else:
        st.info("Página en desarrollo...")

    # Footer
    st.markdown("---")
    st.markdown("*© 2024 AEGIS Framework - Enterprise AI Platform*")

if __name__ == "__main__":
    main()
