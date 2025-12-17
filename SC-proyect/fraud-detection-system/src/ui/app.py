#app.py
from flask import Flask, render_template, jsonify, request, redirect, url_for, flash
import sys
import os
import requests
import json
from datetime import datetime
import pandas as pd
import io
import csv
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_app():
    app = Flask(__name__,
                template_folder='templates',
                static_folder='static',
                static_url_path='/static')
    
    app.config['SECRET_KEY'] = 'fraud-detection-ui-2024'
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['API_BASE_URL'] = 'http://127.0.0.1:5000'  # API de back en puerto 5000
    
    @app.route('/')
    def index():
        return redirect(url_for('dashboard'))
    
    @app.route('/dashboard')
    def dashboard():
        api_status = check_api_status()
        
        return render_template('dashboard.html',
                            page_title='Dashboard - Sistema de Detección de Fraude',
                            active_page='dashboard',
                            api_status=api_status,
                            api_url=app.config['API_BASE_URL'])
    

    
    @app.route('/predict', methods=['GET'])
    def predict_page():
        return render_template('predict.html',
                            page_title='Predicción Masiva',
                            active_page='predict',
                            api_url=app.config['API_BASE_URL'])
    
    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            if 'file' not in request.files:
                return jsonify({
                    'error': True,
                    'message': 'No se seleccionó ningún archivo'
                }), 400
            
            file = request.files['file']
            
            if not file.filename.lower().endswith('.csv'):
                return jsonify({
                    'error': True,
                    'message': 'El archivo debe ser CSV (.csv)'
                }), 400
            
            if file.filename == '':
                return jsonify({
                    'error': True,
                    'message': 'No se seleccionó ningún archivo'
                }), 400
            
            
            content = file.read().decode('utf-8')
            file.seek(0) 
            
            csv_reader = csv.DictReader(io.StringIO(content))
            fieldnames = csv_reader.fieldnames
            
            column_mapping = {
                'Unnamed: 0': 'id_transaccion',
                'amt': 'monto',
                'cc_num': 'tarjeta_credito',
                'city_pop': 'poblacion_ciudad',
                'lat': 'latitud_cliente',
                'long': 'longitud_cliente',
                'merch_lat': 'latitud_comercio',
                'merch_long': 'longitud_comercio',
                'unix_time': 'tiempo_unix',
                'zip': 'codigo_postal'
            }
            
            needs_conversion = False
            if fieldnames:
                english_cols = ['Unnamed: 0', 'amt', 'cc_num']
                needs_conversion = any(col in fieldnames for col in english_cols)
            
            if needs_conversion:
                
                output = io.StringIO()
                
                spanish_columns = []
                for eng_col in fieldnames:
                    if eng_col in column_mapping:
                        spanish_columns.append(column_mapping[eng_col])
                    else:
                        spanish_columns.append(eng_col)
                
                writer = csv.DictWriter(output, fieldnames=spanish_columns)
                writer.writeheader()
                
                for row in csv_reader:
                    new_row = {}
                    for eng_key, value in row.items():
                        if eng_key in column_mapping:
                            esp_key = column_mapping[eng_key]
                            new_row[esp_key] = value
                        else:
                            new_row[eng_key] = value
                    writer.writerow(new_row)
                
                converted_content = output.getvalue()
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as tmp:
                    tmp.write(converted_content)
                    tmp_path = tmp.name
                
                with open(tmp_path, 'rb') as f:
                    files = {'file': (file.filename, f, 'text/csv')}
                    api_url = f"{app.config['API_BASE_URL']}/predict"
                    response = requests.post(api_url, files=files)
                
                os.unlink(tmp_path)
                
            else:
                # si ya es español
                files = {'file': (file.filename, file.stream, file.content_type)}
                api_url = f"http://127.0.0.1:5000/predict"
                response = requests.post(api_url, files=files)
            
            if response.status_code == 200:
                return jsonify(response.json()), 200
            else:
                return jsonify({
                    'error': True,
                    'message': f'Error en la API: {response.status_code}',
                    'details': response.text[:500] if response.text else 'Sin detalles'
                }), response.status_code
                
        except requests.exceptions.ConnectionError:
            return jsonify({
                'error': True,
                'message': 'No se pudo conectar con la API de predicción',
                'details': f'Verifica que la API esté corriendo en {app.config["API_BASE_URL"]}'
            }), 503
            
        except requests.exceptions.Timeout:
            return jsonify({
                'error': True,
                'message': 'La API tardó demasiado en responder',
                'details': 'Timeout de conexión'
            }), 504
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"[ERROR] {error_details}")
            return jsonify({
                'error': True,
                'message': 'Error al procesar el archivo',
                'details': str(e),
                'traceback': error_details[:500]
            }), 500
        
    @app.route('/about')
    def about_page():
        return render_template('about.html',
                            page_title='Sobre nosotros',
                            active_page='about',
                            api_url=app.config['API_BASE_URL'])
    
    @app.route('/api/predict_manual', methods=['POST'])
    def predict_manual_api():
        try:
            #datos JSON
            if not request.is_json:
                return jsonify({
                    'error': True,
                    'message': 'Content-Type debe ser application/json'
                }), 400
            
            data = request.get_json()
            
            if not data:
                return jsonify({
                    'error': True,
                    'message': 'No se enviaron datos'
                }), 400
            
            column_mapping = {
                'Unnamed: 0': 'id_transaccion',
                'amt': 'monto',
                'cc_num': 'tarjeta_credito',
                'city_pop': 'poblacion_ciudad',
                'lat': 'latitud_cliente',
                'long': 'longitud_cliente',
                'merch_lat': 'latitud_comercio',
                'merch_long': 'longitud_comercio',
                'unix_time': 'tiempo_unix',
                'zip': 'codigo_postal'
            }
            
            data_es = {}
            for key_eng, value in data.items():
                if key_eng in column_mapping:
                    key_es = column_mapping[key_eng]
                    data_es[key_es] = value
                else:
                    data_es[key_eng] = value
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as tmp:
                writer = csv.DictWriter(tmp, fieldnames=data_es.keys())
                writer.writeheader()
                writer.writerow(data_es)
                tmp_path = tmp.name
            
            api_url = f"{app.config['API_BASE_URL']}/predict"
            with open(tmp_path, 'rb') as f:
                files = {'file': ('manual_prediction.csv', f, 'text/csv')}
                response = requests.post(api_url, files=files)
            
            os.unlink(tmp_path)
            
            if response.status_code == 200:
                api_result = response.json()
                
                if api_result.get('predicciones') and len(api_result['predicciones']) > 0:
                    primera_pred = api_result['predicciones'][0]
                    
                    return jsonify({
                        'is_fraud': primera_pred['es_fraude'],
                        'probability': primera_pred['probabilidad_fraude'],
                        'risk_level': primera_pred['nivel_riesgo'],
                        'confidence': primera_pred.get('confianza_modelo', 0.95),
                        'threshold': primera_pred.get('umbral_decision', 0.5),
                        'transaction_id': primera_pred.get('id_transaccion', data.get('Unnamed: 0', 0)),
                        'amount': primera_pred.get('monto', data.get('amt', 0)),
                        'raw_api_response': api_result  # Opcional: enviar respuesta completa
                    })
                else:
                    return jsonify({
                        'error': True,
                        'message': 'La API no devolvió predicciones',
                        'api_response': api_result
                    }), 500
            else:
                return jsonify({
                    'error': True,
                    'message': f'Error en la API: {response.status_code}',
                    'details': response.text[:200] if response.text else 'Sin detalles'
                }), response.status_code
                
        except Exception as e:
            return jsonify({
                'error': True,
                'message': 'Error interno del servidor',
                'details': str(e)
            }), 500 
    
    @app.route('/manual_predict', methods=['GET', 'POST'])
    def manual_predict():
        result = None
        error = None
        
        if request.method == 'POST':
            try:
                transaction_data = {
                    "id_transaccion": request.form.get('id_transaccion'),
                    "monto": float(request.form.get('monto', 0)),
                    "tarjeta_credito": request.form.get('tarjeta_credito'),
                    "poblacion_ciudad": int(request.form.get('poblacion_ciudad', 0)),
                    "latitud_cliente": float(request.form.get('latitud_cliente', 0)),
                    "longitud_cliente": float(request.form.get('longitud_cliente', 0)),
                    "latitud_comercio": float(request.form.get('latitud_comercio', 0)),
                    "longitud_comercio": float(request.form.get('longitud_comercio', 0)),
                    "tiempo_unix": int(request.form.get('tiempo_unix', int(datetime.now().timestamp()))),
                    "codigo_postal": request.form.get('codigo_postal')
                }
                
                # archivpo CSV temporal para enviar a la API
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
                    writer = csv.DictWriter(tmp, fieldnames=transaction_data.keys())
                    writer.writeheader()
                    writer.writerow(transaction_data)
                    tmp_path = tmp.name
                
                api_url = f"{app.config['API_BASE_URL']}/predict"
                with open(tmp_path, 'rb') as f:
                    files = {'file': ('manual_predict.csv', f, 'text/csv')}
                    response = requests.post(api_url, files=files)
                
                os.unlink(tmp_path)
                
                if response.status_code == 200:
                    result = response.json()
                else:
                    error = f"Error API: {response.status_code} - {response.text}"
                    
            except Exception as e:
                error = str(e)
        
        return render_template('manual_predict.html',
                            page_title='Predicción Manual',
                            active_page='manual_predict',
                            result=result,
                            error=error,
                            current_time=int(datetime.now().timestamp()))
    
    @app.route('/api/health')
    def api_health():
        # verifica estado de la API
        try:
            response = requests.get(f"{app.config['API_BASE_URL']}/", timeout=5)
            return jsonify({
                'api_status': 'online' if response.status_code == 200 else 'error',
                'status_code': response.status_code,
                'timestamp': datetime.now().isoformat()
            })
        except requests.exceptions.ConnectionError:
            return jsonify({
                'api_status': 'offline',
                'message': 'No se puede conectar con la API',
                'timestamp': datetime.now().isoformat()
            }), 503
    
    
    
    @app.route('/about')
    def about():
        """Página acerca del proyecto."""
        return render_template('dashboard.html',  # Temporal, misma que dashboard
                            page_title='Acerca del Proyecto',
                            active_page='about')
    
    def make_report_pdf():
        """Generar reporte PDF (función placeholder)."""
        pass  # Implementar generación de PDF si es necesario
    
    def check_api_status():
        """Verificar estado de la API."""
        try:
            response = requests.get(f"{app.config['API_BASE_URL']}/", timeout=2)
            return {
                'status': 'online',
                'code': response.status_code,
                'message': 'API conectada correctamente'
            }
        except requests.exceptions.ConnectionError:
            return {
                'status': 'offline',
                'code': 0,
                'message': f'No se puede conectar con la API en {app.config["API_BASE_URL"]}'
            }
        except Exception as e:
            return {
                'status': 'error',
                'code': 0,
                'message': str(e)
            }
    
    @app.route('/geo_visualization/<transaction_id>')
    def geo_visualization(transaction_id):
        """Visualización geográfica de una transacción específica."""
        return render_template('geo_visualization.html',
                            page_title='Visualización Geográfica',
                            active_page='predict',
                            transaction_id=transaction_id,
                            api_url=app.config['API_BASE_URL'])

    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('dashboard.html',
                            error_message='Página no encontrada',
                            active_page='dashboard'), 404
    
    @app.errorhandler(500)
    def internal_server_error(e):
        return render_template('dashboard.html',
                            error_message='Error interno del servidor',
                            active_page='dashboard'), 500
    
    return app

if __name__ == '__main__':
    app = create_app()
    print("-" * 60)
    print(f"Sistema en:    http://localhost:5001/dashboard")
    print("API en: python3 -m src.api.app")
    print("-" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5001)