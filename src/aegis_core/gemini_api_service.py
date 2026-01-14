"""
AEGIS Open AGI - Gemini API Service
Servicio REST API para integración de Gemini AI
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import json
import logging
from datetime import datetime
from gemini_integration import GeminiIntegration, initialize_gemini

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear aplicación Flask
app = Flask(__name__)
CORS(app)

# Inicializar Gemini
try:
    gemini = initialize_gemini()
    logger.info("Gemini AI inicializado correctamente")
except Exception as e:
    logger.error(f"Error inicializando Gemini: {str(e)}")
    gemini = None

# Template HTML para la interfaz web
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AEGIS - Gemini AI Interface</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .header h1 {
            color: #333;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            color: #666;
            font-size: 1.1em;
        }
        
        .chat-container {
            display: grid;
            grid-template-columns: 1fr 300px;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .chat-area {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            height: 400px;
            overflow-y: auto;
            border: 1px solid #e9ecef;
        }
        
        .controls {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            border: 1px solid #e9ecef;
        }
        
        .message {
            margin-bottom: 15px;
            padding: 10px 15px;
            border-radius: 10px;
            max-width: 80%;
        }
        
        .user-message {
            background: #007bff;
            color: white;
            margin-left: auto;
        }
        
        .ai-message {
            background: #e9ecef;
            color: #333;
        }
        
        .input-group {
            display: flex;
            gap: 10px;
            margin-top: 20px;
        }
        
        .input-group input {
            flex: 1;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
        }
        
        .btn {
            padding: 12px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        
        .btn-primary {
            background: #007bff;
            color: white;
        }
        
        .btn-primary:hover {
            background: #0056b3;
        }
        
        .btn-secondary {
            background: #6c757d;
            color: white;
            margin-top: 10px;
            width: 100%;
        }
        
        .btn-secondary:hover {
            background: #545b62;
        }
        
        .model-select {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        
        .status {
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 15px;
            text-align: center;
        }
        
        .status.success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .status.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        .loading {
            display: none;
            text-align: center;
            color: #666;
        }
        
        @media (max-width: 768px) {
            .chat-container {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 AEGIS - Gemini AI</h1>
            <p>Interfaz de Google Gemini AI integrada en el framework AEGIS</p>
        </div>
        
        <div id="status" class="status" style="display: none;"></div>
        
        <div class="chat-container">
            <div class="chat-area" id="chatArea">
                <div class="message ai-message">
                    ¡Hola! Soy Gemini AI integrado en AEGIS. ¿En qué puedo ayudarte hoy?
                </div>
            </div>
            
            <div class="controls">
                <label for="modelSelect">Modelo:</label>
                <select id="modelSelect" class="model-select">
                    <option value="gemini-1.5-flash">Gemini 1.5 Flash (Rápido)</option>
                    <option value="gemini-1.5-pro">Gemini 1.5 Pro (Avanzado)</option>
                    <option value="gemini-pro">Gemini Pro</option>
                </select>
                
                <button class="btn btn-secondary" onclick="clearChat()">Limpiar Chat</button>
                <button class="btn btn-secondary" onclick="getModelInfo()">Info Modelos</button>
            </div>
        </div>
        
        <div class="input-group">
            <input type="text" id="messageInput" placeholder="Escribe tu mensaje aquí..." onkeypress="handleKeyPress(event)">
            <button class="btn btn-primary" onclick="sendMessage()">Enviar</button>
        </div>
        
        <div class="loading" id="loading">
            <p>🤔 Gemini está pensando...</p>
        </div>
    </div>

    <script>
        let conversationHistory = [];
        
        function showStatus(message, type) {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = `status ${type}`;
            status.style.display = 'block';
            setTimeout(() => {
                status.style.display = 'none';
            }, 3000);
        }
        
        function addMessage(content, isUser = false) {
            const chatArea = document.getElementById('chatArea');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user-message' : 'ai-message'}`;
            messageDiv.textContent = content;
            chatArea.appendChild(messageDiv);
            chatArea.scrollTop = chatArea.scrollHeight;
        }
        
        function showLoading(show) {
            document.getElementById('loading').style.display = show ? 'block' : 'none';
        }
        
        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            const model = document.getElementById('modelSelect').value;
            
            if (!message) return;
            
            // Agregar mensaje del usuario
            addMessage(message, true);
            conversationHistory.push({role: 'user', content: message});
            
            // Limpiar input y mostrar loading
            input.value = '';
            showLoading(true);
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        messages: conversationHistory,
                        model: model
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    addMessage(data.response);
                    conversationHistory.push({role: 'model', content: data.response});
                    showStatus('Respuesta generada correctamente', 'success');
                } else {
                    addMessage(`Error: ${data.error}`);
                    showStatus(`Error: ${data.error}`, 'error');
                }
            } catch (error) {
                addMessage(`Error de conexión: ${error.message}`);
                showStatus('Error de conexión', 'error');
            } finally {
                showLoading(false);
            }
        }
        
        async function getModelInfo() {
            showLoading(true);
            
            try {
                const response = await fetch('/api/models');
                const data = await response.json();
                
                if (data.success) {
                    const info = `Modelos disponibles: ${data.models.length}\\n` +
                               `Modelo actual: ${data.default_model}`;
                    addMessage(info);
                    showStatus('Información de modelos obtenida', 'success');
                } else {
                    addMessage(`Error obteniendo modelos: ${data.error}`);
                    showStatus('Error obteniendo información', 'error');
                }
            } catch (error) {
                addMessage(`Error: ${error.message}`);
                showStatus('Error de conexión', 'error');
            } finally {
                showLoading(false);
            }
        }
        
        function clearChat() {
            document.getElementById('chatArea').innerHTML = 
                '<div class="message ai-message">¡Hola! Soy Gemini AI integrado en AEGIS. ¿En qué puedo ayudarte hoy?</div>';
            conversationHistory = [];
            showStatus('Chat limpiado', 'success');
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
        
        // Enfocar input al cargar
        document.addEventListener('DOMContentLoaded', function() {
            document.getElementById('messageInput').focus();
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Página principal con interfaz de chat"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/generate', methods=['POST'])
def generate_text():
    """
    Endpoint para generar texto con Gemini
    
    Body JSON:
    {
        "prompt": "texto del prompt",
        "model": "nombre del modelo (opcional)",
        "temperature": 0.7,
        "max_tokens": 1000
    }
    """
    try:
        if not gemini:
            return jsonify({
                'success': False,
                'error': 'Gemini no está inicializado'
            }), 500
        
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({
                'success': False,
                'error': 'Prompt requerido'
            }), 400
        
        result = gemini.generate_text(
            prompt=data['prompt'],
            model_name=data.get('model'),
            temperature=data.get('temperature', 0.7),
            max_tokens=data.get('max_tokens', 1000)
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error en /api/generate: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Endpoint para conversación de chat
    
    Body JSON:
    {
        "messages": [
            {"role": "user", "content": "mensaje"},
            {"role": "model", "content": "respuesta"}
        ],
        "model": "nombre del modelo (opcional)"
    }
    """
    try:
        if not gemini:
            return jsonify({
                'success': False,
                'error': 'Gemini no está inicializado'
            }), 500
        
        data = request.get_json()
        
        if not data or 'messages' not in data:
            return jsonify({
                'success': False,
                'error': 'Mensajes requeridos'
            }), 400
        
        result = gemini.chat_conversation(
            messages=data['messages'],
            model_name=data.get('model')
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error en /api/chat: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    """
    Endpoint para análisis de imágenes
    
    Form data:
    - image: archivo de imagen
    - prompt: texto del prompt (opcional)
    """
    try:
        if not gemini:
            return jsonify({
                'success': False,
                'error': 'Gemini no está inicializado'
            }), 500
        
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Imagen requerida'
            }), 400
        
        image_file = request.files['image']
        prompt = request.form.get('prompt', 'Describe esta imagen')
        
        # Guardar imagen temporalmente
        temp_path = f"/tmp/{image_file.filename}"
        image_file.save(temp_path)
        
        # Analizar imagen
        result = gemini.analyze_image(temp_path, prompt)
        
        # Limpiar archivo temporal
        os.remove(temp_path)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error en /api/analyze-image: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """Obtiene información sobre los modelos disponibles"""
    try:
        if not gemini:
            return jsonify({
                'success': False,
                'error': 'Gemini no está inicializado'
            }), 500
        
        result = gemini.get_model_info()
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error en /api/models: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/set-model', methods=['POST'])
def set_model():
    """
    Cambia el modelo por defecto
    
    Body JSON:
    {
        "model": "nombre del modelo"
    }
    """
    try:
        if not gemini:
            return jsonify({
                'success': False,
                'error': 'Gemini no está inicializado'
            }), 500
        
        data = request.get_json()
        
        if not data or 'model' not in data:
            return jsonify({
                'success': False,
                'error': 'Nombre del modelo requerido'
            }), 400
        
        success = gemini.set_model(data['model'])
        
        return jsonify({
            'success': success,
            'message': f"Modelo cambiado a {data['model']}" if success else "Error cambiando modelo"
        })
        
    except Exception as e:
        logger.error(f"Error en /api/set-model: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Obtiene el estado del servicio Gemini"""
    return jsonify({
        'success': True,
        'gemini_initialized': gemini is not None,
        'service': 'AEGIS Gemini API Service',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint no encontrado'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Error interno del servidor'
    }), 500

if __name__ == '__main__':
    # Configurar puerto desde variables de entorno
    port = int(os.getenv('GEMINI_PORT', 8053))
    
    logger.info(f"Iniciando AEGIS Gemini API Service en puerto {port}")
    
    # Ejecutar aplicación
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        threaded=True
    )