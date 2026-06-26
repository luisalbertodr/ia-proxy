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

@app.route('/v1/chat/completions', methods=['POST'])
def proxy():
    try:
        data = request.json
        logger.info(f"Request recibida")
        response = requests.post(LITELLM_URL, json=data)
        result = response.json()
        if 'choices' in result and len(result['choices']) > 0:
            message = result['choices'][0].get('message', {})
            content = message.get('content', '')
            if content:
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
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
