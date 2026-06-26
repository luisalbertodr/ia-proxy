from flask import Flask, request, jsonify
import requests
import json
import uuid
import re
import os
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LITELLM_URL = os.environ.get('LITELLM_URL', 'http://litellm:4000/v1/chat/completions')

def extract_tool_call_from_text(text):
    if not text or not isinstance(text, str):
        return None
    
    json_pattern = r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}'
    matches = re.findall(json_pattern, text)
    for match in matches:
        try:
            parsed = json.loads(match)
            if 'name' in parsed and 'arguments' in parsed:
                name = parsed.get('name') or parsed.get('function_name')
                if name:
                    return {
                        'name': name,
                        'arguments': json.dumps(parsed.get('arguments', {}))
                    }
        except:
            continue
    
    func_pattern = r'\{[^{}]*"function_name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}'
    matches = re.findall(func_pattern, text)
    for match in matches:
        try:
            parsed = json.loads(match)
            if 'function_name' in parsed and 'arguments' in parsed:
                return {
                    'name': parsed['function_name'],
                    'arguments': json.dumps(parsed['arguments'])
                }
        except:
            continue
    
    return None

@app.route('/v1/chat/completions', methods=['POST', 'OPTIONS'])
def proxy():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response
    
    try:
        # Verificar Content-Type
        if not request.is_json:
            logger.warning(f"Content-Type no es JSON: {request.content_type}")
            return jsonify({'error': 'Content-Type must be application/json'}), 415
        
        data = request.json
        if not data:
            logger.warning("Datos JSON vacíos")
            return jsonify({'error': 'Empty JSON data'}), 400
        
        logger.info(f"Request recibida")
        
        if 'model' not in data:
            data['model'] = 'localIA:latest'
        
        # Reenviar a LiteLLM
        try:
            response = requests.post(
                LITELLM_URL,
                json=data,
                headers={'Content-Type': 'application/json'},
                timeout=120
            )
            
            # Verificar que la respuesta no está vacía
            if not response.text or response.text.strip() == '':
                logger.error("Respuesta vacía de LiteLLM")
                return jsonify({'error': 'Empty response from LiteLLM'}), 500
            
            # Verificar que es JSON válido
            try:
                result = response.json()
            except json.JSONDecodeError as e:
                logger.error(f"Respuesta no JSON de LiteLLM: {response.text[:200]}")
                return jsonify({'error': f'Invalid JSON from LiteLLM: {str(e)}'}), 500
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error conectando a LiteLLM: {str(e)}")
            return jsonify({'error': f'Error connecting to LiteLLM: {str(e)}'}), 500
        
        # Verificar si la respuesta contiene un tool call en texto plano
        if 'choices' in result and len(result['choices']) > 0:
            message = result['choices'][0].get('message', {})
            content = message.get('content', '')
            
            if content and isinstance(content, str):
                tool_call = extract_tool_call_from_text(content)
                if tool_call:
                    logger.info(f"Tool call detectado: {tool_call['name']}")
                    result['choices'][0]['message']['content'] = None
                    result['choices'][0]['message']['tool_calls'] = [{
                        'id': f'call_{uuid.uuid4().hex[:8]}',
                        'type': 'function',
                        'function': {
                            'name': tool_call['name'],
                            'arguments': tool_call['arguments']
                        }
                    }]
                    result['choices'][0]['finish_reason'] = 'tool_calls'
        
        json_response = jsonify(result)
        json_response.headers.add('Access-Control-Allow-Origin', '*')
        return json_response
        
    except json.JSONDecodeError as e:
        logger.error(f"Error decodificando JSON: {str(e)}")
        return jsonify({'error': f'Invalid JSON: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        error_response = jsonify({'error': str(e)})
        error_response.headers.add('Access-Control-Allow-Origin', '*')
        return error_response, 500

@app.route('/health', methods=['GET'])
def health():
    response = jsonify({'status': 'ok'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
