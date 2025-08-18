# 🚀 Guía de Despliegue en Streamlit Cloud

## 📋 Pasos para Desplegar

### 1. **Preparar el Repositorio de GitHub**

1. **Crear un nuevo repositorio en GitHub:**
   ```
   https://github.com/new
   ```

2. **Estructura de archivos necesaria:**
   ```
   tu-repositorio/
   ├── streamlit_app.py          # Archivo principal (OBLIGATORIO)
   ├── requirements.txt          # Dependencias
   ├── README.md                # Documentación
   ├── .gitignore               # Archivos a ignorar
   └── .streamlit/
       └── config.toml          # Configuración de Streamlit
   ```

3. **Subir todos los archivos:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit - Product Benchmark Tool"
   git branch -M main
   git remote add origin https://github.com/TU-USUARIO/TU-REPO.git
   git push -u origin main
   ```

### 2. **Configurar Streamlit Cloud**

1. **Ir a Streamlit Cloud:**
   ```
   https://share.streamlit.io/
   ```

2. **Conectar con GitHub:**
   - Inicia sesión con tu cuenta de GitHub
   - Autoriza Streamlit Cloud

3. **Crear nueva app:**
   - Click en "New app"
   - Selecciona tu repositorio
   - Branch: `main`
   - Main file path: `streamlit_app.py`
   - App URL: `tu-app-name` (personalizable)

### 3. **Configuración Avanzada**

#### **Variables de Entorno (Opcional)**
Si necesitas configuraciones especiales, crea `.streamlit/secrets.toml`:

```toml
# .streamlit/secrets.toml
[general]
user_agent = "ProductBenchmarkTool/1.0"
timeout = 15
max_retries = 3

[rate_limiting]
delay_min = 0.5
delay_max = 5.0
```

#### **Configuración de Tema**
El archivo `.streamlit/config.toml` ya está incluido con:
- Tema personalizado
- Configuración de servidor
- Optimizaciones de rendimiento

### 4. **Verificar Dependencias**

El archivo `requirements.txt` incluye todas las librerías necesarias:

```txt
streamlit==1.28.0
requests==2.31.0
beautifulsoup4==4.12.2
pandas==2.0.3
nltk==3.8.1
wordcloud==1.9.2
matplotlib==3.7.2
seaborn==0.12.2
textblob==0.17.1
plotly==5.17.0
lxml==4.9.3
```

### 5. **Proceso de Despliegue**

1. **Automático:** Una vez configurado, Streamlit Cloud:
   - Detecta cambios en GitHub
   - Reinstala dependencias si es necesario
   - Redespliega automáticamente

2. **Tiempo de despliegue:** 2-5 minutos normalmente

3. **URL final:** `https://tu-app-name.streamlit.app`

## 🔧 Solución de Problemas Comunes

### **Error: "Module not found"**
```bash
# Verificar que requirements.txt esté en la raíz
# Verificar nombres exactos de paquetes
# Reiniciar el despliegue desde Streamlit Cloud
```

### **Error: "App is taking too long"**
```python
# Optimizar el código:
# - Usar @st.cache_resource para NLTK
# - Reducir número de requests simultáneos
# - Añadir timeouts apropiados
```

### **Error: "Memory limit exceeded"**
```python
# Optimizaciones:
# - Limitar número de productos analizados
# - Usar menos features en visualizaciones
# - Optimizar uso de pandas DataFrames
```

### **Error con NLTK downloads**
```python
# El código ya incluye manejo de errores para NLTK
# Usar stopwords básicas si NLTK falla
# Verificar conexión a internet en el servidor
```

## 📊 Monitoreo y Mantenimiento

### **Métricas de Uso**
Streamlit Cloud proporciona:
- Número de visitantes
- Tiempo de uso de la app
- Errores y logs

### **Logs de Errores**
- Accesibles desde el panel de Streamlit Cloud
- Útiles para debuggear problemas de usuarios

### **Actualizaciones**
```bash
# Cualquier push a main actualiza automáticamente
git add .
git commit -m "Actualización: nueva funcionalidad"
git push origin main
```

## 🎯 Optimizaciones de Rendimiento

### **Caching Estratégico**
```python
# Ya implementado en el código:
@st.cache_resource
def download_nltk_data():
    # Cachea descargas de NLTK
```

### **Límites Recomendados**
- **URLs simultáneas:** 5-10 (para evitar timeouts)
- **Delay entre requests:** 1-2 segundos
- **Timeout por request:** 15 segundos

### **Gestión de Memoria**
```python
# Usar generators para grandes datasets
# Limpiar variables no usadas
# Limitar tamaño de DataFrames
```

## 🔒 Seguridad y Buenas Prácticas

### **Rate Limiting**
- Implementado delay configurable
- Headers de User-Agent apropiados
- Manejo de errores de conexión

### **Respeto a robots.txt**
```python
# Considerar añadir verificación de robots.txt
# Respetar términos de servicio de sitios web
# No hacer scraping agresivo
```

### **Privacidad**
- No almacenar URLs de usuarios
- No hacer logs de datos sensibles
- Procesar datos en memoria únicamente

## 📞 Soporte y Recursos

### **Documentación Oficial**
- [Streamlit Cloud Docs](https://docs.streamlit.io/streamlit-cloud)
- [Deployment Guide](https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app)

### **Comunidad**
- [Streamlit Forum](https://discuss.streamlit.io/)
- [GitHub Issues](https://github.com/streamlit/streamlit/issues)

### **Límites de Streamlit Cloud (Plan Gratuito)**
- **CPU:** Limitado
- **RAM:** ~1GB
- **Almacenamiento:** Temporal
- **Ancho de banda:** Generoso para uso normal
- **Apps concurrentes:** 3 apps

## 🎉 ¡Listo para Producción!

Una vez desplegado, tu herramienta estará disponible 24/7 en:
```
https://tu-app-name.streamlit.app
```

**Próximos pasos sugeridos:**
1. Probar con diferentes tipos de sitios web
2. Recopilar feedback de usuarios
3. Iterar y mejorar funcionalidades
4. Considerar migrar a plan de pago para más recursos si es necesario

---

**¿Necesitas ayuda?** 
- Revisa los logs en Streamlit Cloud
- Verifica que todos los archivos estén en GitHub
- Asegúrate de que `streamlit_app.py` esté en la raíz del repo
