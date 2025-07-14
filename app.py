import os
import json
import uuid
import base64
import sqlite3
import wave
import io
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import google.generativeai as genai
import pandas as pd
import requests
from google.oauth2 import service_account

# Try to import the speech module correctly
try:
    from google.cloud import speech
except ImportError:
    speech = None
    print("Warning: Google Cloud Speech-to-Text library not installed. `pip install google-cloud-speech`")

app = Flask(__name__)
CORS(app)

# --- Configuration ---
# IMPORTANT: Replace with your actual Gemini API Key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Models
text_and_vision_model = genai.GenerativeModel('gemini-1.5-flash')
text_only_model = genai.GenerativeModel('gemini-1.5-flash')

# File Upload Configuration
UPLOAD_FOLDER = 'uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Database Configuration
DB_FILE = 'knowledge.db'

# Password for adding knowledge
KNOWLEDGE_PASSWORD = "Arquitas de Tarento"

# Speech-to-Text Client
SPEECH_CLIENT = None
try:
    CREDENTIALS_PATH = os.path.join(os.path.dirname(__file__), 'credentials.json')
    if os.path.exists(CREDENTIALS_PATH) and speech:
        credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
        SPEECH_CLIENT = speech.SpeechClient(credentials=credentials)
        print("✅ Google Cloud Speech-to-Text client loaded successfully.")
    else:
        print("⚠️ Speech-to-Text credentials ('credentials.json') not found or library not installed. Audio features will be disabled.")
except Exception as e:
    print(f"❌ Error setting up Speech-to-Text client: {str(e)}")
    SPEECH_CLIENT = None

# --- Database Setup ---
def get_db_connection():
    """Creates a connection to the SQLite database."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database with the provided knowledge base."""
    if os.path.exists(DB_FILE):
        return # Avoid re-initializing

    print("🚀 Initializing new knowledge base database...")
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute('''
    CREATE TABLE knowledge (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category TEXT NOT NULL,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        UNIQUE(category, key)
    )
    ''')

    initial_knowledge = {
        "equipments": {
            "Empilhadeira padrão (\"42\")": json.dumps({"capacity": "Até 7 toneladas", "observation": "Única unidade identificada como '42'."}),
            "Empilhadeira de grande porte": json.dumps({"capacity": "Até 35 toneladas", "observation": "Indicada para Open Top / Flat Rack com cargas pesadas."}),
            "Capacidade de desova manual (CNTR DRAY)": json.dumps({"capacity": "L: 1,20 m / A: 1,55 m / Peso < 7 toneladas", "observation": "Para cargas manuseadas sem equipamento especializado."})
        },
        "desova_types": {
            "Mecanizada": json.dumps({"time": "1 hora", "equipe": "2 OPERADORES, 1 CONFERENTE, 2 AUXILIARES", "criteria": "Cargas paletizadas ou com peso individual elevado"}),
            "Manual": json.dumps({"time": "4 horas", "equipe": "2 OPERADORES, 1 CONFERENTE, 3 AUXILIARES", "criteria": "Cargas leves, soltas ou em caixas"}),
            "Carga Projeto": json.dumps({"time": "> 4 horas", "equipe": "1 OPERADOR, 1 CONFERENTE, 2 AUXILIARES", "criteria": "Cargas > 7 toneladas. Requer uso de empilhadeira de grande porte."}),
            "Híbrida (Mecanizada/Manual)": json.dumps({"criteria": "Desovas de rolos ou rolls com peso unitário > 60kg. Realizada de forma mecanizada com auxílio manual.", "time": "Variável", "equipe": "2 OPERADORES, 1 CONFERENTE, 2 AUXILIARES"})
        },
        "container_types": {
            "Dry Van / Standard": json.dumps({"abreviations": ["DV", "GP", "DC"], "iso_codes": ["22G1", "42G1"]}),
            "High Cube": json.dumps({"abreviations": ["HC", "HQ"], "iso_codes": ["45G1", "L5G1"]}),
            "Reefer": json.dumps({"abreviations": ["RF", "RH"], "iso_codes": ["22R1", "42R1"]}),
            "Open Top": json.dumps({"abreviations": ["OT"], "iso_codes": ["22U1", "42U1"]}),
            "Flat Rack": json.dumps({"abreviations": ["FR"], "iso_codes": ["22P1", "42P1"]}),
            "Platform": json.dumps({"abreviations": ["PF", "PL"], "iso_codes": []}),
            "Tank": json.dumps({"abreviations": ["TK"], "iso_codes": ["22T4"]}),
            "Ventilated": json.dumps({"abreviations": ["VN"], "iso_codes": ["22V0", "42V0"]}),
            "Open Side": json.dumps({"abreviations": ["OS"], "iso_codes": ["22D1", "42D1"]})
        },
        "storage_locations": {
            "Drive-in": json.dumps({"capacity": "1 tonelada por pallet (acima do nível 01)", "observation": "Presente somente na Fila H. Posição '01' (chão) não tem limite de peso."}),
            "Autoportante": json.dumps({"capacity": "1.5 tonelada por pallet (acima do nível 01)", "observation": "Cada fila comporta até 3 pallets por andar. Posição '01' (chão) não tem limite de peso."}),
            "Containers Baú": json.dumps({"capacity": "137 unidades", "use_criteria": "Ocupação > 85% nos sistemas fixos", "advantages": "Agilidade na retirada", "disadvantages": "Tempo adicional para estufagem/vistoria.", "procedure": "Requer vistoria detalhada para danos e infiltrações."})
        },
        "special_cases": {
            "sulfato de amonia": json.dumps({"keywords": ["sulfato de amônia", "sulfato de amonia", "sulfato de amônio", "sulfato de amoníaco", "NH₄SO₄", "sanduicheiras"], "procedure": "Risco de odor forte. Utilizar 4 auxiliares com máscaras faciais obrigatórias e pausas para descanso. Operação pode durar até 5 dias."})
        },
        "operational_rules": {
            "rejection_weight_limit": "Carga com peso unitário superior a 100 quilos não deve ser aprovada para desova por questões de segurança.",
            "manual_viability_weight": "Carga com peso unitário inferior a 35 quilos, se puder ser acondicionada em caixas, a desova manual é viável."
        }
    }

    for category, items in initial_knowledge.items():
        for key, value in items.items():
            cur.execute("INSERT INTO knowledge (category, key, value) VALUES (?, ?, ?)", (category, key, value))

    conn.commit()
    conn.close()
    print("✅ Database initialized with default knowledge.")


def get_knowledge_base():
    """Retrieves the entire knowledge base from the database and formats it as a dict."""
    conn = get_db_connection()
    rows = conn.execute("SELECT category, key, value FROM knowledge").fetchall()
    conn.close()
    
    kb = {}
    for row in rows:
        if row['category'] not in kb:
            kb[row['category']] = {}
        try:
            kb[row['category']][row['key']] = json.loads(row['value'])
        except json.JSONDecodeError:
            kb[row['category']][row['key']] = row['value']
            
    return kb

# --- Helper Functions ---
def get_ncm_description(ncm_code):
    """Fetches NCM description from an external API."""
    try:
        ncm_clean = "".join(filter(str.isdigit, ncm_code))
        response = requests.get(f"https://brasilapi.com.br/api/ncm/v1/{ncm_clean}")
        if response.status_code == 200:
            return response.json().get("descricao", "Descrição não disponível")
        return "Erro na consulta NCM"
    except Exception as e:
        return f"Falha na consulta NCM: {str(e)}"

# --- API Routes ---
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index_test.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo recebido'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nenhum arquivo selecionado'}), 400

    filename = file.filename
    file_ext = os.path.splitext(filename)[1].lower()
    
    # Gerar um nome de arquivo temporário para salvar
    temp_filename = str(uuid.uuid4()) + file_ext
    file_path = os.path.join(UPLOAD_FOLDER, temp_filename)
    file.save(file_path)

    contents_for_gemini = []
    prompt_text = ""
    extracted_text = ""
    
    try:
        if file_ext in ['.jpg', '.jpeg', '.png']:
            mime_type = f"image/{file_ext[1:]}"
            img_part = {"mime_type": mime_type, "data": base64.b64encode(open(file_path, "rb").read()).decode()}
            contents_for_gemini.append(img_part)
            prompt_text = "Esta é uma imagem de um documento de carga (BL, Pack-List, FISPQ). Analise-a criticamente."
            response = text_and_vision_model.generate_content(["Extraia todo o texto desta imagem.", img_part])
            extracted_text = response.text

        elif file_ext == '.pdf':
            mime_type = 'application/pdf'
            pdf_part = {"mime_type": mime_type, "data": base64.b64encode(open(file_path, "rb").read()).decode()}
            contents_for_gemini.append(pdf_part)
            prompt_text = "Este é um documento PDF de carga (BL, Pack-List, FISPQ). Analise-o criticamente."
            response = text_and_vision_model.generate_content(["Extraia todo o texto deste PDF.", pdf_part])
            extracted_text = response.text
        
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, header=None)
            excel_data_as_text = df.to_json(orient='values', indent=2)
            prompt_text = f"Analise os dados desta planilha Excel de carga:\n{excel_data_as_text}"
            contents_for_gemini.append(prompt_text)
            extracted_text = excel_data_as_text

        else:
            return jsonify({'error': 'Formato de arquivo não suportado'}), 400

        knowledge_base = get_knowledge_base()

        system_prompt = f"""
        Você é 'Morpheus', um assistente especialista em análise crítica operacional para o Grupo Armazém Salvador.
        Seu objetivo é ler documentos de carga, identificar informações chave e sugerir um plano operacional com base no conhecimento interno.

        Sua base de conhecimento: {json.dumps(knowledge_base, indent=2, ensure_ascii=False)}

        Instruções:
        1. Analise o documento fornecido.
        2. Extraia as seguintes informações: tipo de container, código ISO, se é carga perigosa (IMO), NCM, peso total, nome do produto e quantidade de pacotes/volumes.
        3. Com base nas características da carga e na base de conhecimento, sugira: o tipo de desova (Manual, Mecanizada, etc.), o equipamento necessário, a equipe recomendada e o local de armazenamento.
        4. Identifique e liste quaisquer alertas importantes (ex: carga perigosa, peso unitário excedendo limites, casos especiais como sulfato de amônia).
        5. Calcule o peso unitário (peso total / quantidade) se possível.
        6. Forneça uma justificativa clara para suas sugestões.
        7. Retorne APENAS um objeto JSON válido com a seguinte estrutura:
        {{
          "container_type": "string",
          "iso_code": "string",
          "imo_detected": boolean,
          "ncm": "string",
          "weight": "string",
          "product": "string",
          "packages": "string",
          "peso_unitario": "string",
          "desova_suggestion": "string",
          "equipamento_suggestion": "string",
          "equipe_suggestion": "string",
          "armazenamento_suggestion": "string",
          "tempo_estimado": "string",
          "alerts": ["string"],
          "justification": "string"
        }}
        """
        
        contents_for_gemini.insert(0, system_prompt)
        contents_for_gemini.insert(1, prompt_text)

        response = text_and_vision_model.generate_content(contents_for_gemini)
        ai_response_content = response.text.strip()
        
        if ai_response_content.startswith('```json'):
            ai_response_content = ai_response_content[7:]
        if ai_response_content.endswith('```'):
            ai_response_content = ai_response_content[:-3]
        
        ai_analysis_result = json.loads(ai_response_content)
        
        if ai_analysis_result.get('ncm'):
            ai_analysis_result['ncm_description'] = get_ncm_description(ai_analysis_result['ncm'])

        doc_id = str(uuid.uuid4())
        
        if 'DOCUMENT_CONTEXT_DB' not in app.config:
            app.config['DOCUMENT_CONTEXT_DB'] = {}
        app.config['DOCUMENT_CONTEXT_DB'][doc_id] = {
            'filename': filename,
            'text_content': extracted_text
        }
        
        return jsonify({
            'arquivo': filename,
            'tipo': file_ext,
            'analise': ai_analysis_result,
            'doc_id': doc_id
        })

    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        print(f"Received content from AI: {ai_response_content}")
        return jsonify({'error': f'Falha ao interpretar a resposta da IA. Resposta recebida: {ai_response_content}'}), 500
    except Exception as e:
        print(f"Error in /upload: {e}")
        return jsonify({'error': f'Falha na análise: {str(e)}'}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    doc_id = data.get('doc_id')
    chat_history = data.get('history', [])
    
    doc_context_info = ""
    if doc_id and 'DOCUMENT_CONTEXT_DB' in app.config and doc_id in app.config['DOCUMENT_CONTEXT_DB']:
        doc_info = app.config['DOCUMENT_CONTEXT_DB'][doc_id]
        doc_context_info = f"""
        Contexto do Documento Ativo:
        - Nome do Arquivo: {doc_info.get('filename')}
        - Conteúdo Extraído (resumo): {doc_info.get('text_content', '')[:1500]}...
        """

    try:
        knowledge_base = get_knowledge_base()
        system_prompt = f"""
        Você é 'Morpheus', o assistente de análise de cargas do Grupo Armazém Salvador.
        Seja prestativo, informativo e direto. Responda em português brasileiro.
        Use a base de conhecimento e o contexto do documento ativo para responder.
        {doc_context_info}
        Base de Conhecimento Atual: {json.dumps(knowledge_base, indent=2, ensure_ascii=False)}
        """
        
        model_history = []
        for msg in chat_history:
            role = 'user' if msg['role'] == 'user' else 'model'
            model_history.append({'role': role, 'parts': [msg['content']]})

        model_history.insert(0, {'role': 'model', 'parts': [system_prompt]})

        chat_session = text_only_model.start_chat(history=model_history)
        response = chat_session.send_message(user_message)
        
        return jsonify({'response': response.text})

    except Exception as e:
        print(f"Error in /chat: {e}")
        return jsonify({'response': f'Desculpe, ocorreu um erro no servidor: {str(e)}'})


@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    if not SPEECH_CLIENT:
        return jsonify({'error': 'Serviço de reconhecimento de fala não configurado no servidor.'}), 501
    
    if 'audio' not in request.files:
        return jsonify({'error': 'Nenhum áudio recebido'}), 400
    
    audio_file = request.files['audio']
    audio_content = audio_file.read()
    filename = audio_file.filename
    
    ENCODING_MAP = {
        '.wav': speech.RecognitionConfig.AudioEncoding.LINEAR16,
        '.flac': speech.RecognitionConfig.AudioEncoding.FLAC,
        '.ogg': speech.RecognitionConfig.AudioEncoding.OGG_OPUS,
    }

    file_extension = os.path.splitext(filename)[1].lower() if filename else None
    encoding = ENCODING_MAP.get(file_extension)

    config_dict = {
        "language_code": "pt-BR",
        "enable_automatic_punctuation": True,
        "model": "latest_short"
    }

    try:
        if encoding:
            print(f"Detectado formato de áudio: {file_extension}. Usando codificação: {encoding.name}")
            config_dict["encoding"] = encoding
            if encoding == speech.RecognitionConfig.AudioEncoding.LINEAR16:
                try:
                    audio_buffer = io.BytesIO(audio_content)
                    with wave.open(audio_buffer, 'rb') as wav_file:
                        sample_rate = wav_file.getframerate()
                        print(f"Taxa de amostragem lida do arquivo WAV: {sample_rate}")
                        config_dict["sample_rate_hertz"] = sample_rate
                except wave.Error as e:
                    print(f"Erro ao ler o cabeçalho do WAV: {e}")
                    return jsonify({'error': f'Não foi possível ler o arquivo WAV: {e}'}), 400
        else:
            print("Nenhum formato conhecido detectado. Usando configuração padrão para WEBM/Opus.")
            config_dict["encoding"] = speech.RecognitionConfig.AudioEncoding.WEBM_OPUS
            config_dict["sample_rate_hertz"] = 48000

        config = speech.RecognitionConfig(**config_dict)
        audio = speech.RecognitionAudio(content=audio_content)
        
        response = SPEECH_CLIENT.recognize(config=config, audio=audio)
        transcript = " ".join([result.alternatives[0].transcript for result in response.results])

        return jsonify({'text': transcript.strip()})
    
    except Exception as e:
        print(f"Speech-to-text error: {e}")
        return jsonify({'error': f'Falha no reconhecimento de fala: {str(e)}'}), 500

# --- ROTA INTELIGENTE PARA ADICIONAR CONHECIMENTO ---
@app.route('/add-knowledge-smart', methods=['POST'])
def add_knowledge_smart():
    data = request.get_json()
    category = data.get('category')
    key = data.get('key')
    user_text = data.get('text')

    if not all([category, key, user_text]):
        return jsonify({'error': 'Categoria, Chave e Texto são obrigatórios.'}), 400

    prompt = f"""
    Sua tarefa é atuar como um especialista em estruturação de dados para um sistema de logística.
    Um usuário forneceu as seguintes informações em linguagem natural para serem adicionadas à base de conhecimento na categoria '{category}' sob a chave '{key}'.

    Texto do usuário: "{user_text}"

    Instruções:
    1. Analise o texto do usuário para extrair as informações essenciais.
    2. Corrija quaisquer erros de digitação e melhore o texto para um tom técnico e formal.
    3. Estruture essas informações em um formato JSON válido.
    4. Se a informação for um simples valor de texto (como em 'operational_rules'), o JSON deve ser apenas uma string entre aspas.
    5. Se a informação tiver múltiplos atributos (como em 'equipments' ou 'special_cases'), crie um objeto JSON com pares de chave-valor.
    6. Retorne APENAS o JSON estruturado, sem nenhum outro texto, explicação ou markdown.

    Exemplo para um equipamento: Se o usuário digitar "nova empilhadeira, aguenta 10 ton, eletrica", o resultado deve ser:
    {{"capacity": "Até 10 toneladas", "observation": "Empilhadeira elétrica."}}

    Exemplo para uma regra: Se o usuário digitar "nao pode paletizar acima de 2 metros", o resultado deve ser:
    "Proibida a paletização com altura superior a 2.0 metros por questões de segurança."

    Agora, processe o texto do usuário e gere o JSON correspondente.
    """

    try:
        response = text_only_model.generate_content(prompt)
        json_response = response.text.strip()
        if '```json' in json_response:
             json_response = json_response.split('```json', 1)[1].rsplit('```', 1)[0].strip()

        json.loads(json_response)
        
        return jsonify({'structured_json': json_response})

    except Exception as e:
        print(f"Erro na geração de JSON pela IA: {e}")
        return jsonify({'error': f'A IA não conseguiu estruturar a informação. Tente ser mais específico. Erro: {str(e)}'}), 500


# --- ROTA ORIGINAL PARA SALVAR CONHECIMENTO ---
@app.route('/add-knowledge', methods=['POST'])
def add_knowledge():
    data = request.get_json()
    password = data.get('password')
    category = data.get('category')
    key = data.get('key')
    value = data.get('value')

    if password != KNOWLEDGE_PASSWORD:
        return jsonify({'error': 'Senha incorreta!'}), 403

    if not all([category, key, value]):
        return jsonify({'error': 'Categoria, Chave e Valor são campos obrigatórios.'}), 400

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO knowledge (category, key, value) VALUES (?, ?, ?)
            ON CONFLICT(category, key) DO UPDATE SET value=excluded.value
        """, (category, key, value))
        conn.commit()
        conn.close()
        return jsonify({'success': f'Conhecimento na categoria "{category}" foi adicionado/atualizado com sucesso!'})
    except Exception as e:
        return jsonify({'error': f'Erro no banco de dados: {str(e)}'}), 500


if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5001, debug=True)