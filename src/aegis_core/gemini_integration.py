"""
AEGIS Open AGI - Gemini AI Integration Module
Integración de Google Gemini AI para el framework AEGIS
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
import google.generativeai as genai
from datetime import datetime

class GeminiIntegration:
    """
    Clase principal para la integración de Google Gemini AI
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Inicializa la integración de Gemini
        
        Args:
            api_key: Clave API de Google Gemini (opcional, se puede cargar desde variables de entorno)
        """
        self.logger = logging.getLogger(__name__)
        
        # Configurar API key
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv('GEMINI_API_KEY')
            
        if not self.api_key:
            raise ValueError("API key de Gemini no encontrada. Configura GEMINI_API_KEY en variables de entorno.")
            
        # Configurar Gemini
        genai.configure(api_key=self.api_key)
        
        # Modelos disponibles
        self.models = {
            'gemini-pro': 'gemini-pro',
            'gemini-pro-vision': 'gemini-pro-vision',
            'gemini-1.5-pro': 'gemini-1.5-pro',
            'gemini-1.5-flash': 'gemini-1.5-flash'
        }
        
        # Modelo por defecto
        self.default_model = 'gemini-1.5-flash'
        self.model = genai.GenerativeModel(self.default_model)
        
        self.logger.info(f"Gemini AI integrado exitosamente con modelo: {self.default_model}")
    
    def generate_text(self, prompt: str, model_name: Optional[str] = None, 
                     temperature: float = 0.7, max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Genera texto usando Gemini AI
        
        Args:
            prompt: Texto de entrada para generar respuesta
            model_name: Nombre del modelo a usar (opcional)
            temperature: Creatividad de la respuesta (0.0 - 1.0)
            max_tokens: Máximo número de tokens en la respuesta
            
        Returns:
            Dict con la respuesta generada y metadatos
        """
        try:
            # Seleccionar modelo
            if model_name and model_name in self.models:
                model = genai.GenerativeModel(self.models[model_name])
            else:
                model = self.model
            
            # Configurar generación
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            
            # Generar respuesta
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            result = {
                'success': True,
                'text': response.text,
                'model': model_name or self.default_model,
                'timestamp': datetime.now().isoformat(),
                'prompt_tokens': len(prompt.split()),
                'completion_tokens': len(response.text.split()) if response.text else 0
            }
            
            self.logger.info(f"Texto generado exitosamente con {result['completion_tokens']} tokens")
            return result
            
        except Exception as e:
            self.logger.error(f"Error generando texto: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def analyze_image(self, image_path: str, prompt: str = "Describe esta imagen") -> Dict[str, Any]:
        """
        Analiza una imagen usando Gemini Pro Vision
        
        Args:
            image_path: Ruta a la imagen
            prompt: Prompt para el análisis
            
        Returns:
            Dict con el análisis de la imagen
        """
        try:
            import PIL.Image
            
            # Cargar imagen
            image = PIL.Image.open(image_path)
            
            # Usar modelo de visión
            vision_model = genai.GenerativeModel('gemini-pro-vision')
            
            # Generar análisis
            response = vision_model.generate_content([prompt, image])
            
            result = {
                'success': True,
                'analysis': response.text,
                'image_path': image_path,
                'prompt': prompt,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Imagen analizada exitosamente: {image_path}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error analizando imagen: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def chat_conversation(self, messages: List[Dict[str, str]], 
                         model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Mantiene una conversación de chat con Gemini
        
        Args:
            messages: Lista de mensajes [{'role': 'user'/'model', 'content': 'texto'}]
            model_name: Modelo a usar
            
        Returns:
            Dict con la respuesta del chat
        """
        try:
            # Seleccionar modelo
            if model_name and model_name in self.models:
                model = genai.GenerativeModel(self.models[model_name])
            else:
                model = self.model
            
            # Iniciar chat
            chat = model.start_chat(history=[])
            
            # Procesar mensajes previos
            for msg in messages[:-1]:
                if msg['role'] == 'user':
                    chat.send_message(msg['content'])
            
            # Enviar último mensaje y obtener respuesta
            last_message = messages[-1]['content']
            response = chat.send_message(last_message)
            
            result = {
                'success': True,
                'response': response.text,
                'conversation_length': len(messages),
                'model': model_name or self.default_model,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Conversación procesada con {len(messages)} mensajes")
            return result
            
        except Exception as e:
            self.logger.error(f"Error en conversación: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtiene información sobre los modelos disponibles
        
        Returns:
            Dict con información de modelos
        """
        try:
            available_models = []
            for model in genai.list_models():
                if 'generateContent' in model.supported_generation_methods:
                    available_models.append({
                        'name': model.name,
                        'display_name': model.display_name,
                        'description': model.description,
                        'input_token_limit': model.input_token_limit,
                        'output_token_limit': model.output_token_limit
                    })
            
            return {
                'success': True,
                'models': available_models,
                'default_model': self.default_model,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error obteniendo información de modelos: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def set_model(self, model_name: str) -> bool:
        """
        Cambia el modelo por defecto
        
        Args:
            model_name: Nombre del modelo
            
        Returns:
            True si el cambio fue exitoso
        """
        try:
            if model_name in self.models:
                self.model = genai.GenerativeModel(self.models[model_name])
                self.default_model = model_name
                self.logger.info(f"Modelo cambiado a: {model_name}")
                return True
            else:
                self.logger.error(f"Modelo no disponible: {model_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error cambiando modelo: {str(e)}")
            return False

# Funciones de utilidad para integración con AEGIS
def initialize_gemini(config_path: str = None) -> GeminiIntegration:
    """
    Inicializa Gemini con configuración desde archivo
    
    Args:
        config_path: Ruta al archivo de configuración
        
    Returns:
        Instancia de GeminiIntegration
    """
    try:
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            api_key = config.get('gemini_api_key')
        else:
            api_key = None
            
        return GeminiIntegration(api_key=api_key)
        
    except Exception as e:
        logging.error(f"Error inicializando Gemini: {str(e)}")
        raise

def create_gemini_config(api_key: str, config_path: str = "config/gemini_config.json"):
    """
    Crea archivo de configuración para Gemini
    
    Args:
        api_key: Clave API de Gemini
        config_path: Ruta donde guardar la configuración
    """
    try:
        config = {
            'gemini_api_key': api_key,
            'default_model': 'gemini-1.5-flash',
            'temperature': 0.7,
            'max_tokens': 1000,
            'created_at': datetime.now().isoformat()
        }
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Guardar configuración
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        logging.info(f"Configuración de Gemini guardada en: {config_path}")
        
    except Exception as e:
        logging.error(f"Error creando configuración: {str(e)}")
        raise

if __name__ == "__main__":
    # Ejemplo de uso
    try:
        # Inicializar Gemini
        gemini = GeminiIntegration()
        
        # Generar texto
        result = gemini.generate_text("Explica qué es la inteligencia artificial")
        print("Respuesta:", result['text'] if result['success'] else result['error'])
        
        # Obtener información de modelos
        models_info = gemini.get_model_info()
        print("Modelos disponibles:", len(models_info['models']) if models_info['success'] else "Error")
        
    except Exception as e:
        print(f"Error en ejemplo: {str(e)}")