from flask import Flask, request, jsonify, Response, stream_with_context
import requests
import json
import uuid
import os
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LITELLM_URL = os.environ.get('LITELLM_URL', 'http://localhost:4000/v1/chat/completions')
LITELLM_API_KEY = os.environ.get('LITELLM_API_KEY', 'sk-local-placeholder-key')

# Alias aceptados para cada campo, en orden de prioridad
NAME_KEYS = ['name', 'function_name', 'tool_name', 'Name']
ARGS_KEYS = ['arguments', 'function_arguments', 'args', 'parameters', 'input']


def find_json_objects(text):
    """
    Encuentra todos los objetos JSON de nivel superior en un texto,
    balanceando llaves correctamente (soporta cualquier nivel de anidamiento,
    a diferencia de un regex con [^{}]*).
    """
    objects = []
    depth = 0
    start = None
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    objects.append(text[start:i + 1])
                    start = None

    return objects


def extract_tool_call_from_text(text):
    if not text or not isinstance(text, str):
        return None

    for candidate in find_json_objects(text):
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue

        if not isinstance(parsed, dict):
            continue

        # Si viene envuelto en { "function": {...} } / { "Function": {...} }
        # o en el formato OpenAI { "id":..., "type":"function", "function": {...} }
        inner = parsed
        for fk in ('function', 'Function'):
            if fk in parsed and isinstance(parsed[fk], dict):
                inner = parsed[fk]
                break

        name = next((inner[k] for k in NAME_KEYS if k in inner and inner[k]), None)
        args = next((inner[k] for k in ARGS_KEYS if k in inner), None)

        if name and args is not None:
            args_json = args if isinstance(args, str) else json.dumps(args)
            logger.info(f"Tool call detectado vía claves: name={name}, args_key usado")
            return {'name': name, 'arguments': args_json}

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

        # Tool calling nativo: ya NO se elimina 'tools' ni 'tool_choice'.
        # Se reenvían tal cual a LiteLLM/Ollama para que el modelo (qwen2.5-coder,
        # que soporta function calling) genere tool_calls estructurados de forma
        # nativa, usando el TEMPLATE del Modelfile ({{ if .Tools }} / {{ if .ToolCalls }}).
        has_tools = bool(data.get('tools'))
        if has_tools:
            logger.info(f"Petición con {len(data['tools'])} tool(s) definidas — modo nativo")

        if 'max_tokens' not in data:
            data['max_tokens'] = 16384

        is_stream = data.get('stream', False)

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {LITELLM_API_KEY}'
        }
        if is_stream:
            headers['Accept'] = 'text/event-stream'

        if is_stream:
            def generate():
                accumulated_content = ""
                saw_native_tool_call = False
                buffered_lines = []

                try:
                    response = requests.post(
                        LITELLM_URL, json=data, headers=headers,
                        timeout=120, stream=True
                    )

                    if response.status_code != 200:
                        error_msg = f"Error from LiteLLM: {response.status_code} - {response.text[:200]}"
                        logger.error(error_msg)
                        yield f"data: {json.dumps({'error': error_msg})}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    for line in response.iter_lines():
                        if not line:
                            continue
                        decoded = line.decode('utf-8')

                        if decoded == 'data: [DONE]':
                            # Fin del stream: si no hubo tool_call nativo, intentamos
                            # el fallback sobre el contenido acumulado, y si algo se
                            # rescata, lo inyectamos como un chunk extra de tool_calls
                            # antes de cerrar, para que Cursor lo procese igualmente.
                            if not saw_native_tool_call and accumulated_content.strip():
                                logger.info(
                                    f"Sin tool_calls nativos en streaming. "
                                    f"Contenido acumulado a analizar: {accumulated_content[:500]}"
                                )
                                tool_call = extract_tool_call_from_text(accumulated_content)
                                if tool_call:
                                    logger.warning(
                                        f"⚠️ Tool call rescatado por fallback de texto "
                                        f"(streaming, no nativo): {tool_call['name']}. "
                                        f"Revisa el Modelfile/plantilla si esto ocurre con frecuencia."
                                    )
                                    fallback_chunk = {
                                        'id': f'chatcmpl-{uuid.uuid4().hex[:8]}',
                                        'object': 'chat.completion.chunk',
                                        'choices': [{
                                            'index': 0,
                                            'delta': {
                                                'tool_calls': [{
                                                    'index': 0,
                                                    'id': f'call_{uuid.uuid4().hex[:8]}',
                                                    'type': 'function',
                                                    'function': {
                                                        'name': tool_call['name'],
                                                        'arguments': tool_call['arguments']
                                                    }
                                                }]
                                            },
                                            'finish_reason': 'tool_calls'
                                        }]
                                    }
                                    yield f"data: {json.dumps(fallback_chunk)}\n\n"
                                else:
                                    logger.info("❌ No se detectó tool call ni nativo ni por texto (streaming)")
                            elif saw_native_tool_call:
                                logger.info("✅ Tool call nativo recibido (streaming)")

                            yield "data: [DONE]\n\n"
                            continue

                        if decoded.startswith('data: '):
                            # Inspeccionamos el chunk para acumular content y detectar tool_calls nativos
                            try:
                                chunk_json = json.loads(decoded[len('data: '):])
                                choices = chunk_json.get('choices', [])
                                if choices:
                                    delta = choices[0].get('delta', {})
                                    if delta.get('tool_calls'):
                                        saw_native_tool_call = True
                                    piece = delta.get('content')
                                    if piece:
                                        accumulated_content += piece
                            except Exception:
                                pass  # chunk no parseable como JSON, se reenvía igual

                            yield f"{decoded}\n\n"

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
            try:
                response = requests.post(
                    LITELLM_URL, json=data, headers=headers, timeout=120
                )

                if response.status_code != 200:
                    logger.error(f"LiteLLM error: {response.status_code} - {response.text[:200]}")
                    return jsonify({'error': f'LiteLLM error: {response.status_code}'}), response.status_code

                result = response.json()

                if 'choices' in result and len(result['choices']) > 0:
                    message = result['choices'][0].get('message', {})
                    content = message.get('content', '')
                    native_tool_calls = message.get('tool_calls')

                    if native_tool_calls:
                        # El modelo ya generó tool_calls estructurados de forma nativa
                        # (vía .Tools/.ToolCalls del Modelfile). No tocar nada.
                        logger.info(f"✅ Tool call nativo recibido: "
                                    f"{[tc.get('function', {}).get('name') for tc in native_tool_calls]}")

                    elif content and isinstance(content, str):
                        logger.info(f"Sin tool_calls nativos. Contenido a analizar: {content[:500]}")

                        # Fallback: el modelo respondió en texto plano en vez de
                        # usar el mecanismo nativo. Se intenta rescatar igualmente.
                        tool_call = extract_tool_call_from_text(content)

                        if tool_call:
                            logger.warning(
                                f"⚠️ Tool call rescatado por fallback de texto (no nativo): "
                                f"{tool_call['name']}. Revisa el Modelfile/plantilla si esto "
                                f"ocurre con frecuencia."
                            )
                            logger.info(f"Arguments: {tool_call['arguments']}")

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
                            logger.info("❌ No se detectó tool call ni nativo ni por texto, respuesta normal")

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
