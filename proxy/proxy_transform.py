from flask import Flask, request, jsonify, Response, stream_with_context
import requests
import json
import uuid
import re
import os
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LITELLM_URL = os.environ.get('LITELLM_URL', 'http://localhost:4000/v1/chat/completions')

def extract_tool_call_from_text(text):
    if not text or not isinstance(text, str):
        return None
    
    # Múltiples patrones para detectar tool calls en texto plano
    patterns = [
        # Formato: { "id": "call_1", "type": "function", "Function": { "name": "...", "arguments": {...} } }
        r'\{[^{}]*"id"\s*:\s*"[^"]+"\s*,\s*"type"\s*:\s*"[^"]+"\s*,\s*"[Ff]unction"\s*:\s*\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}\}',
        # Formato: { "function": { "name": "...", "arguments": {...} } }
        r'\{[^{}]*"function"\s*:\s*\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}\}',
        # Formato: { "name": "...", "arguments": {...} }
        r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}',
        # Formato: { "function_name": "...", "function_arguments": {...} }
        r'\{[^{}]*"function_name"\s*:\s*"[^"]+"\s*,\s*"function_arguments"\s*:\s*\{[^{}]*\}[^{}]*\}'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                parsed = json.loads(match)
                logger.info(f"Intentando parsear: {parsed}")
                
                # Formato: { "id": "...", "type": "function", "Function": { ... } }
                # o { "id": "...", "type": "function", "function": { ... } }
                if 'id' in parsed and 'type' in parsed and parsed['type'] == 'function':
                    func_key = 'Function' if 'Function' in parsed else 'function'
                    if func_key in parsed and isinstance(parsed[func_key], dict):
                        func = parsed[func_key]
                        if 'name' in func and 'arguments' in func:
                            return {
                                'name': func['name'],
                                'arguments': json.dumps(func['arguments'])
                            }
                
                # Formato: { "function": { "name": "...", "arguments": {...} } }
                if 'function' in parsed and isinstance(parsed['function'], dict):
                    func = parsed['function']
                    if 'name' in func and 'arguments' in func:
                        return {
                            'name': func['name'],
                            'arguments': json.dumps(func['arguments'])
                        }
                
                # Formato: { "name": "...", "arguments": {...} }
                if 'name' in parsed and 'arguments' in parsed:
                    return {
                        'name': parsed['name'],
                        'arguments': json.dumps(parsed['arguments'])
                    }
                
                # Formato: { "function_name": "...", "function_arguments": {...} }
                if 'function_name' in parsed and 'function_arguments' in parsed:
                    return {
                        'name': parsed['function_name'],
                        'arguments': json.dumps(parsed['function_arguments'])
                    }
                    
            except Exception as e:
                logger.debug(f"Error parseando match: {str(e)}")
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
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 415
        
        data = request.json
        logger.info(f"Request recibida")
        
        if 'model' not in data:
            data['model'] = 'localIA:latest'
        
        # Eliminar tools para evitar error en LiteLLM
        if 'tools' in data:
            del data['tools']
        
        # Forzar max_tokens si no está presente
        if 'max_tokens' not in data:
            data['max_tokens'] = 16384
        
        is_stream = data.get('stream', False)
        
        # Headers para LiteLLM
        headers = {
            'Content-Type': 'application/json'
        }
        
        # Si es streaming, añadir Accept correcto
        if is_stream:
            headers['Accept'] = 'text/event-stream'
        
        if is_stream:
            # Manejar streaming
            def generate():
                try:
                    response = requests.post(
                        LITELLM_URL,
                        json=data,
                        headers=headers,
                        timeout=120,
                        stream=True
                    )
                    
                    if response.status_code != 200:
                        error_msg = f"Error from LiteLLM: {response.status_code} - {response.text[:200]}"
                        logger.error(error_msg)
                        yield f"data: {json.dumps({'error': error_msg})}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    
                    # Reenviar el stream línea por línea
                    for line in response.iter_lines():
                        if line:
                            decoded = line.decode('utf-8')
                            # Reenviar solo líneas que comienzan con "data: "
                            if decoded.startswith('data: '):
                                yield f"{decoded}\n\n"
                            # También reenviar líneas "data: [DONE]"
                            elif decoded == 'data: [DONE]':
                                yield "data: [DONE]\n\n"
                    
                except Exception as e:
                    logger.error(f"Error en streaming: {str(e)}")
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                    yield "data: [DONE]\n\n"
            
            return Response(
                stream_with_context(generate()),
                status=200,
                headers={
                    'Content-Type': 'text/event-stream',
                    'Cache-Control': 'no-cache',
                    'Access-Control-Allow-Origin': '*'
                }
            )
        else:
            # Manejar respuesta no streaming
            try:
                response = requests.post(
                    LITELLM_URL,
                    json=data,
                    headers=headers,
                    timeout=120
                )
                
                if response.status_code != 200:
                    logger.error(f"LiteLLM error: {response.status_code} - {response.text[:200]}")
                    return jsonify({'error': f'LiteLLM error: {response.status_code}'}), response.status_code
                
                result = response.json()
                
                # Verificar tool calls en la respuesta
                if 'choices' in result and len(result['choices']) > 0:
                    message = result['choices'][0].get('message', {})
                    content = message.get('content', '')
                    
                    if content and isinstance(content, str):
                        logger.info(f"Contenido a analizar: {content[:500]}")
                        
                        # Buscar tool call en el contenido
                        tool_call = extract_tool_call_from_text(content)
                        
                        if tool_call:
                            logger.info(f"✅ Tool call detectado: {tool_call['name']}")
                            logger.info(f"Arguments: {tool_call['arguments']}")
                            
                            # Reemplazar el contenido con el tool call
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
                        else:
                            logger.info("❌ No se detectó tool call, respuesta normal")
                
                json_response = jsonify(result)
                json_response.headers.add('Access-Control-Allow-Origin', '*')
                return json_response
                
            except Exception as e:
                logger.error(f"Error: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
    except Exception as e:
        logger.error(f"Error general: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    response = jsonify({'status': 'ok'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
