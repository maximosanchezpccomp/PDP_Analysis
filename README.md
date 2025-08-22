# 🎯 Análisis Competitivo de Productos

## 📋 Descripción

Herramienta avanzada de análisis competitivo que extrae y compara información de productos desde múltiples sitios web, identificando oportunidades de mejora y gaps en tu estrategia de producto.

## ✨ Características Principales

### 🆕 Nuevas Funcionalidades (v2.0)

- **🎯 Análisis de GAPS**: Compara tu producto contra la competencia
- **📊 Detección de Oportunidades**: Identifica características que tienes vs competencia
- **💰 Análisis de Precios**: Posicionamiento comparativo en el mercado
- **📈 Visualizaciones Avanzadas**: Heatmaps, gráficos radar, comparaciones visuales
- **💾 Exportación Mejorada**: CSV completo + análisis de gaps en TXT

### 🔧 Funcionalidades Core

- **🔗 Extracción Multi-sitio**: Analiza productos de diferentes tiendas online
- **🔤 Análisis de Términos**: Identifica palabras clave más relevantes
- **🎛️ Análisis de Filtros**: Descubre qué filtros usa la competencia
- **⭐ Análisis de Características**: Extrae features más mencionadas
- **🛒 Google Shopping**: Análisis de mercado sin restricciones
- **📊 Visualizaciones Interactivas**: Gráficos con Plotly
- **☁️ Nube de Palabras**: Visualización de términos clave

## 🚀 Despliegue Rápido en Streamlit Cloud

### 1. Fork o Clona el Repositorio

```bash
git clone https://github.com/maximosanchezpccomp/PDP_Analysis.git
cd tu-repo
```

### 2. Estructura de Archivos Requerida

```
PDP_Anlysis/
├── streamlit_app.py          # Archivo principal (OBLIGATORIO)
├── requirements.txt          # Dependencias
├── README.md                # Este archivo
├── .gitignore              # Archivos a ignorar
└── .streamlit/
    └── config.toml         # Configuración de Streamlit
```

### 3. Subir a GitHub

```bash
git add .
git commit -m "feat: análisis competitivo v2.0 con análisis de gaps"
git push origin main
```

### 4. Desplegar en Streamlit Cloud

1. Ve a [share.streamlit.io](https://share.streamlit.io/)
2. Conecta tu cuenta de GitHub
3. Selecciona tu repositorio
4. Branch: `main`
5. Main file: `streamlit_app.py`
6. Click en "Deploy"

### 5. URL de la App

```
[https://tu-app-name.streamlit.app] (https://pdpanalysis.streamlit.app/)
```

## 💻 Instalación Local

### Prerrequisitos

- Python 3.8 o superior
- pip o conda

### Instalación

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/tu-repo.git
cd tu-repo

# Crear entorno virtual (opcional pero recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar aplicación
streamlit run streamlit_app.py
```

## 📖 Cómo Usar

### 🎯 Flujo de Trabajo Recomendado

1. **URL de Referencia** (Opcional): Ingresa la URL de TU producto para análisis de gaps
2. **URLs de Competencia**: Añade URLs de productos competidores
3. **Configuración**: Ajusta parámetros en la barra lateral
4. **Análisis**: Click en "INICIAR ANÁLISIS"
5. **Resultados**: Revisa las diferentes pestañas con insights
6. **Exportar**: Descarga CSV con datos completos o TXT con análisis de gaps

### 📊 Tipos de Análisis Disponibles

#### Análisis de GAPS (NUEVO)
- Características únicas de la competencia
- Especificaciones faltantes en tu producto
- Filtros adicionales que usa la competencia
- Comparación de precios y posicionamiento

#### Análisis Tradicional
- **Términos**: Palabras clave más relevantes
- **Filtros**: Categorías y filtros más comunes
- **Características**: Features más mencionadas
- **Precios**: Análisis de rangos y promedios
- **Visualizaciones**: Gráficos interactivos y comparaciones

### 🛒 Google Shopping

Alternativa para análisis de mercado sin restricciones:
1. Ingresa término de búsqueda
2. Selecciona número de resultados
3. Analiza distribución por tiendas, precios y términos

## ⚙️ Configuración Avanzada

### Opciones Anti-detección

- **🔄 Reintentar bloqueados**: Reintenta URLs que fallan
- **🚀 Modo agresivo**: Delays más largos para sitios difíciles
- **🔄 Rotar User-Agents**: Cambia headers entre requests

### Parámetros Ajustables

- **Top N resultados**: Cantidad de elementos a mostrar (5-50)
- **Delay entre requests**: Tiempo de espera (0.5-5.0 segundos)

## 🌐 Compatibilidad de Sitios

### ✅ Alta Compatibilidad
- Amazon (amazon.es, amazon.com)
- eBay (ebay.es, ebay.com)
- AliExpress
- Tiendas pequeñas/medianas

### ⚠️ Compatibilidad Media
- Grandes marketplaces
- Tiendas con protección básica

### 🚫 Baja Compatibilidad
- MediaMarkt
- PCComponentes
- El Corte Inglés
- Sitios con Cloudflare/protección avanzada

## 📊 Estructura de Datos Extraídos

```python
{
    'url': str,              # URL original
    'domain': str,           # Dominio del sitio
    'title': str,            # Título del producto
    'description': str,      # Descripción
    'features': list,        # Lista de características
    'specifications': dict,  # Especificaciones técnicas
    'price': str,           # Precio encontrado
    'filters': list,        # Filtros disponibles
    'categories': list,     # Categorías/breadcrumbs
    'images': list,         # URLs de imágenes
    'extracted_at': str     # Timestamp de extracción
}
```

## 🔧 Solución de Problemas

### Error: "Module not found"
```bash
pip install --upgrade -r requirements.txt
```

### Error: "Acceso denegado" / 403
- Activa modo agresivo
- Aumenta delay entre requests
- Prueba con sitios alternativos
- Usa Google Shopping como alternativa

### La app es lenta
- Reduce número de URLs simultáneas
- Usa menos opciones de análisis
- Considera usar caché del navegador

## 🚀 Optimización de Rendimiento

### Para Streamlit Cloud

- **Límites del plan gratuito**:
  - CPU: Limitado
  - RAM: ~1GB
  - Ideal para: <20 URLs simultáneas

### Recomendaciones

1. Analiza en lotes de 5-10 URLs
2. Usa delays de 2-3 segundos
3. Activa solo análisis necesarios
4. Exporta datos para análisis offline

## 📈 Casos de Uso

1. **E-commerce**: Análisis de competencia directa
2. **Marketing**: Identificación de keywords y tendencias
3. **Producto**: Detección de features faltantes
4. **Pricing**: Estrategia de precios competitiva
5. **SEO**: Optimización de contenido basada en competencia

## 🤝 Contribuir

Las contribuciones son bienvenidas:

1. Fork el proyecto
2. Crea tu feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push al branch (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo licencia MIT. Ver `LICENSE` para más información.

## 🆘 Soporte

- 📧 Email: maximo.sanchez@pccomponentes.com
- 🐛 Issues: [GitHub Issues](https://github.com/maximosanchezpccomp/PDP_analysis/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/maximosanchezpccomp/PDP_analysis/discussions)

## 🙏 Agradecimientos

- Streamlit por la plataforma
- BeautifulSoup por el parsing HTML
- Plotly por las visualizaciones
- NLTK por el procesamiento de texto

---

**Última actualización**: v2.0 - Análisis de GAPS y mejoras de UI/UX

⭐ Si te resulta útil, considera dar una estrella al repositorio!
