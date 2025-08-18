import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from collections import Counter
import re
from urllib.parse import urlparse, urljoin
import time
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings

# Importar wordcloud de forma opcional
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    st.warning("⚠️ WordCloud no está disponible. Las nubes de palabras estarán deshabilitadas.")

# Importar textblob de forma opcional
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

# Suprimir advertencias
warnings.filterwarnings('ignore')

# Configuración inicial de la página
st.set_page_config(
    page_title="Análisis de Competencia - Productos",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Descargar recursos de NLTK si es necesario
@st.cache_resource
def download_nltk_data():
    """Descarga datos de NLTK necesarios"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

download_nltk_data()

class ProductBenchmarkAnalyzer:
    def __init__(self):
        try:
            self.stop_words = set(nltk.corpus.stopwords.words('spanish') + 
                                 nltk.corpus.stopwords.words('english'))
        except:
            # Si falla NLTK, usar conjunto básico de stopwords
            self.stop_words = set(['el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 
                                 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 
                                 'para', 'al', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 
                                 'to', 'for', 'of', 'with', 'by'])
        self.results = []
        
    def extract_content_from_url(self, url):
        """Extrae contenido relevante de una URL de producto"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'es-ES,es;q=0.8,en-US;q=0.5,en;q=0.3',
                'Accept-Encoding': 'gzip, deflate',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extraer información del producto
            product_data = {
                'url': url,
                'title': self._extract_title(soup),
                'description': self._extract_description(soup),
                'features': self._extract_features(soup),
                'specifications': self._extract_specifications(soup),
                'price': self._extract_price(soup),
                'filters': self._extract_filters(soup),
                'categories': self._extract_categories(soup)
            }
            
            return product_data
            
        except requests.exceptions.RequestException as e:
            st.warning(f"⚠️ Error de conexión con {url[:50]}...: {str(e)}")
            return None
        except Exception as e:
            st.warning(f"⚠️ Error procesando {url[:50]}...: {str(e)}")
            return None
    
    def _extract_title(self, soup):
        """Extrae el título del producto"""
        selectors = [
            'h1[class*="title"]',
            'h1[class*="product"]',
            '[data-testid*="title"]',
            '[class*="product-title"]',
            '[class*="product-name"]',
            '[id*="title"]',
            'h1',
            'title'
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text().strip()
                if text and len(text) > 5 and len(text) < 300:
                    return text
        return ""
    
    def _extract_description(self, soup):
        """Extrae la descripción del producto"""
        selectors = [
            '[class*="description"]',
            '[class*="product-description"]',
            '[class*="summary"]',
            '[class*="overview"]',
            '[class*="details"]',
            '[data-testid*="description"]',
            'meta[name="description"]',
            '[class*="content"]'
        ]
        
        description = ""
        for selector in selectors:
            if 'meta' in selector:
                element = soup.select_one(selector)
                if element:
                    desc = element.get('content', '')
                    if desc and len(desc) > 20:
                        description += desc + " "
            else:
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text().strip()
                    if text and len(text) > 20 and len(text) < 2000:
                        description += text + " "
        
        return description.strip()
    
    def _extract_features(self, soup):
        """Extrae características y features del producto"""
        features = []
        
        # Buscar listas de características
        feature_selectors = [
            '[class*="feature"] li',
            '[class*="benefit"] li',
            '[class*="highlight"] li',
            '[class*="spec"] li',
            '[class*="bullet"] li',
            'ul[class*="feature"] li',
            'ul[class*="list"] li',
            '.features li',
            '.benefits li',
            'div[class*="feature"]',
            'span[class*="feature"]'
        ]
        
        for selector in feature_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text().strip()
                # Filtrar características válidas
                if (text and 
                    len(text) > 10 and 
                    len(text) < 500 and 
                    not re.match(r'^\d+$', text) and  # No solo números
                    not text.lower().startswith(('http', 'www', 'mailto'))):  # No enlaces
                    features.append(text)
        
        # Eliminar duplicados manteniendo orden
        seen = set()
        unique_features = []
        for feature in features:
            if feature.lower() not in seen:
                seen.add(feature.lower())
                unique_features.append(feature)
        
        return unique_features[:50]  # Limitar a 50 características
    
    def _extract_specifications(self, soup):
        """Extrae especificaciones técnicas"""
        specs = {}
        
        # Buscar tablas de especificaciones
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    key = cells[0].get_text().strip()
                    value = cells[1].get_text().strip()
                    if key and value and len(key) < 100 and len(value) < 200:
                        specs[key] = value
        
        # Buscar definiciones
        dts = soup.find_all('dt')
        for dt in dts:
            dd = dt.find_next_sibling('dd')
            if dd:
                key = dt.get_text().strip()
                value = dd.get_text().strip()
                if key and value and len(key) < 100 and len(value) < 200:
                    specs[key] = value
        
        # Buscar pares clave-valor en divs
        spec_divs = soup.find_all(['div', 'span'], class_=re.compile(r'spec|attribute|property'))
        for div in spec_divs:
            text = div.get_text().strip()
            if ':' in text:
                parts = text.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    if key and value and len(key) < 100 and len(value) < 200:
                        specs[key] = value
        
        return specs
    
    def _extract_price(self, soup):
        """Extrae información de precio"""
        price_selectors = [
            '[class*="price"]',
            '[class*="cost"]',
            '[class*="amount"]',
            '[data-testid*="price"]',
            '[id*="price"]',
            '.price',
            '.cost',
            '.amount'
        ]
        
        for selector in price_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text().strip()
                # Buscar patrones de precio más amplios
                price_patterns = [
                    r'[€$£¥]\s*[\d,]+\.?\d*',
                    r'[\d,]+\.?\d*\s*[€$£¥]',
                    r'[\d,]+\.?\d*\s*EUR?',
                    r'[\d,]+\.?\d*\s*USD?',
                    r'[\d,]+\.?\d*\s*euros?',
                    r'[\d,]+\.?\d*\s*dólares?'
                ]
                
                for pattern in price_patterns:
                    price_match = re.search(pattern, text, re.IGNORECASE)
                    if price_match:
                        return price_match.group().strip()
        
        return ""
    
    def _extract_filters(self, soup):
        """Extrae filtros disponibles en la página"""
        filters = []
        
        filter_selectors = [
            '[class*="filter"] a',
            '[class*="facet"] a',
            '[class*="refine"] a',
            '[class*="category"] a',
            'select option',
            '[type="checkbox"] + label',
            '[class*="tag"]',
            '[class*="badge"]',
            'nav a',
            '.filter-option',
            '.facet-option'
        ]
        
        for selector in filter_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text().strip()
                if (text and 
                    len(text) > 2 and 
                    len(text) < 80 and
                    not text.lower().startswith(('http', 'www')) and
                    not re.match(r'^\d+, text)):
                    filters.append(text)
        
        # Eliminar duplicados
        return list(set(filters))[:100]  # Limitar a 100 filtros
    
    def _extract_categories(self, soup):
        """Extrae categorías del producto"""
        categories = []
        
        category_selectors = [
            '[class*="breadcrumb"] a',
            '[class*="category"] a',
            '[class*="nav"] a',
            'nav[class*="breadcrumb"] a',
            '.breadcrumb a',
            '.category a',
            'ol.breadcrumb a'
        ]
        
        for selector in category_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text().strip()
                if (text and 
                    text.lower() not in ['home', 'inicio', 'tienda', 'shop', 'store'] and
                    len(text) > 2 and 
                    len(text) < 50):
                    categories.append(text)
        
        return categories
    
    def analyze_terms(self, all_data):
        """Analiza los términos más frecuentes"""
        all_text = ""
        
        for data in all_data:
            all_text += f" {data.get('title', '')} {data.get('description', '')} "
            all_text += " ".join(data.get('features', []))
            all_text += " ".join(data.get('specifications', {}).keys())
            all_text += " ".join(data.get('specifications', {}).values())
        
        # Limpiar y tokenizar texto
        # Incluir caracteres acentuados y ñ
        words = re.findall(r'\b[a-záéíóúñüA-ZÁÉÍÓÚÑÜ]{3,}\b', all_text.lower())
        words = [word for word in words if word not in self.stop_words]
        
        return Counter(words)
    
    def analyze_filters(self, all_data):
        """Analiza los filtros más comunes"""
        all_filters = []
        
        for data in all_data:
            all_filters.extend(data.get('filters', []))
        
        return Counter(all_filters)
    
    def analyze_features(self, all_data):
        """Analiza las características más mencionadas"""
        all_features = []
        
        for data in all_data:
            all_features.extend(data.get('features', []))
        
        # Extraer palabras clave de las características
        feature_words = []
        for feature in all_features:
            words = re.findall(r'\b[a-záéíóúñüA-ZÁÉÍÓÚÑÜ]{3,}\b', feature.lower())
            words = [word for word in words if word not in self.stop_words]
            feature_words.extend(words)
        
        return Counter(feature_words)

def main():
    # CSS personalizado para mejorar la apariencia
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 3px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 0.75rem 1.25rem;
        margin-bottom: 1rem;
        color: #155724;
    }
    .stProgress .st-bo {
        background-color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header principal
    st.markdown('<h1 class="main-header">📊 Herramienta de Análisis de Competencia</h1>', unsafe_allow_html=True)
    st.markdown("### Analiza fichas de productos de la competencia para obtener insights clave")
    
    # Información de ayuda
    with st.expander("ℹ️ ¿Cómo funciona esta herramienta?"):
        st.markdown("""
        **Esta herramienta te permite:**
        
        1. 🔗 **Analizar múltiples URLs de productos** de diferentes sitios web
        2. 📊 **Extraer automáticamente** títulos, descripciones, características y precios
        3. 🔍 **Identificar patrones** en términos, filtros y features más comunes
        4. 📈 **Visualizar los resultados** con gráficos interactivos
        5. 💾 **Exportar los datos** en formato CSV para análisis adicionales
        
        **Ejemplos de sitios compatibles:**
        - Amazon, eBay, MercadoLibre
        - Tiendas online de retail
        - Sitios web de fabricantes
        - Catálogos de productos B2B
        
        **Consejos para mejores resultados:**
        - Usa URLs de productos específicos (no categorías)
        - Incluye productos similares del mismo nicho
        - Espera unos segundos entre análisis para evitar bloqueos
        """)
    
    # Sidebar para configuración
    st.sidebar.header("⚙️ Configuración del Análisis")
    st.sidebar.markdown("---")
    
    # Opciones de análisis
    st.sidebar.subheader("📋 Tipos de Análisis")
    analyze_terms = st.sidebar.checkbox("🔤 Términos más repetidos", value=True, 
                                       help="Analiza las palabras más frecuentes en descripciones y títulos")
    analyze_filters = st.sidebar.checkbox("🎛️ Filtros más usados", value=True,
                                        help="Identifica los filtros de navegación más comunes")
    analyze_features = st.sidebar.checkbox("⭐ Características más mencionadas", value=True,
                                         help="Extrae las features más destacadas")
    
    # Solo mostrar opción de wordcloud si está disponible
    if WORDCLOUD_AVAILABLE:
        show_wordcloud = st.sidebar.checkbox("☁️ Nube de palabras", value=True,
                                           help="Genera visualización de nube de palabras")
    else:
        show_wordcloud = False
        st.sidebar.info("☁️ Nube de palabras no disponible")
    
    st.sidebar.markdown("---")
    
    # Configuración de límites
    st.sidebar.subheader("🎯 Configuración de Resultados")
    top_n = st.sidebar.slider("📊 Top N resultados", 5, 50, 20, 
                             help="Número de elementos principales a mostrar en cada análisis")
    
    # Configuración de scraping
    st.sidebar.subheader("🔧 Configuración Avanzada")
    delay = st.sidebar.slider("⏱️ Delay entre requests (seg)", 0.5, 5.0, 1.0, 0.5,
                             help="Tiempo de espera entre peticiones para evitar bloqueos")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("💡 **Tip:** Comienza con 3-5 URLs para probar la herramienta")
    
    # Input de URLs
    st.header("🔗 URLs de Productos a Analizar")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        urls_input = st.text_area(
            "Introduce las URLs de productos (una por línea):",
            height=200,
            placeholder="""https://www.amazon.es/producto-ejemplo-1
https://www.pccomponentes.com/producto-ejemplo-2
https://www.mediamarkt.es/producto-ejemplo-3
https://www.elcorteingles.es/producto-ejemplo-4""",
            help="Pega aquí las URLs completas de los productos que quieres analizar"
        )
    
    with col2:
        st.markdown("**📝 Ejemplos de URLs válidas:**")
        st.code("amazon.es/dp/B08X...")
        st.code("pccomponentes.com/...")
        st.code("mediamarkt.es/es/...")
        st.markdown("**⚠️ Evita URLs de:**")
        st.markdown("- Categorías")
        st.markdown("- Búsquedas")
        st.markdown("- Páginas principales")
    
    # Validación básica de URLs
    if urls_input.strip():
        urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
        valid_urls = []
        invalid_urls = []
        
        for url in urls:
            if url.startswith(('http://', 'https://')):
                valid_urls.append(url)
            else:
                invalid_urls.append(url)
        
        if invalid_urls:
            st.warning(f"⚠️ Se encontraron {len(invalid_urls)} URLs que no comienzan con http:// o https://")
            with st.expander("Ver URLs inválidas"):
                for url in invalid_urls:
                    st.text(f"❌ {url}")
        
        if valid_urls:
            st.success(f"✅ {len(valid_urls)} URLs válidas detectadas")
    
    # Botón principal de análisis
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🚀 INICIAR ANÁLISIS COMPLETO", 
                                 type="primary", 
                                 use_container_width=True,
                                 help="Comenzar el análisis de todas las URLs introducidas")
    
    if analyze_button:
        if not urls_input.strip():
            st.error("❌ Por favor, introduce al menos una URL")
            return
        
        urls = [url.strip() for url in urls_input.split('\n') if url.strip() and url.startswith(('http://', 'https://'))]
        
        if not urls:
            st.error("❌ No se encontraron URLs válidas")
            return
        
        analyzer = ProductBenchmarkAnalyzer()
        
        # Contenedor para el progreso
        progress_container = st.container()
        
        with progress_container:
            st.markdown("### 🔄 Procesando URLs...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Métricas en tiempo real
            col1, col2, col3 = st.columns(3)
            with col1:
                success_metric = st.metric("✅ Exitosos", 0)
            with col2:
                failed_metric = st.metric("❌ Fallidos", 0)
            with col3:
                total_metric = st.metric("📊 Total", len(urls))
        
        all_data = []
        failed_count = 0
        
        # Procesar cada URL
        for i, url in enumerate(urls):
            status_text.markdown(f'🔍 **Procesando URL {i+1}/{len(urls)}**  \n`{url[:70]}{"..." if len(url) > 70 else ""}`')
            
            data = analyzer.extract_content_from_url(url)
            if data:
                all_data.append(data)
                success_metric.metric("✅ Exitosos", len(all_data))
            else:
                failed_count += 1
                failed_metric.metric("❌ Fallidos", failed_count)
            
            progress_bar.progress((i + 1) / len(urls))
            time.sleep(delay)  # Usar delay configurado
        
        status_text.markdown('✅ **Análisis completado exitosamente**')
        
        if not all_data:
            st.error("❌ No se pudo extraer información de ninguna URL. Por favor verifica que las URLs sean válidas y accesibles.")
            st.info("💡 **Posibles causas:**\n- URLs incorrectas o inaccesibles\n- Sitios web con protección anti-scraping\n- Problemas de conexión a internet")
            return
        
        # Mostrar mensaje de éxito
        success_msg = f"""
        <div class="success-message">
            <strong>🎉 ¡Análisis completado!</strong><br>
            Se procesaron <strong>{len(all_data)}</strong> de <strong>{len(urls)}</strong> productos exitosamente
            {f'({len(urls) - len(all_data)} fallaron)' if len(all_data) < len(urls) else ''}
        </div>
        """
        st.markdown(success_msg, unsafe_allow_html=True)
        
        # Crear pestañas para los resultados
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Resumen", 
            "🔤 Términos", 
            "🎛️ Filtros", 
            "⭐ Características", 
            "📈 Visualizaciones"
        ])
        
        with tab1:
            st.header("📊 Resumen del Análisis")
            
            # Métricas principales
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🔗 Productos Analizados", len(all_data))
            
            with col2:
                total_features = sum(len(data.get('features', [])) for data in all_data)
                st.metric("⭐ Total Características", total_features)
            
            with col3:
                total_specs = sum(len(data.get('specifications', {})) for data in all_data)
                st.metric("🔧 Total Especificaciones", total_specs)
            
            with col4:
                products_with_price = sum(1 for data in all_data if data.get('price'))
                st.metric("💰 Con Precio", products_with_price)
            
            st.markdown("---")
            
            # Crear tabla resumen
            summary_data = []
            for i, data in enumerate(all_data):
                domain = urlparse(data['url']).netloc
                summary_data.append({
                    '#': i + 1,
                    'Dominio': domain,
                    'Título': data['title'][:60] + '...' if len(data['title']) > 60 else data['title'],
                    'Precio': data['price'] or 'N/A',
                    'Características': len(data['features']),
                    'Especificaciones': len(data['specifications']),
                    'Filtros': len(data['filters'])
                })
            
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, use_container_width=True, hide_index=True)
            
            # Mostrar algunos productos de ejemplo
            if len(all_data) > 0:
                st.subheader("🔍 Vista Previa de Productos")
                
                selected_product = st.selectbox(
                    "Selecciona un producto para ver detalles:",
                    options=range(len(all_data)),
                    format_func=lambda x: f"{x+1}. {all_data[x]['title'][:50]}..." if len(all_data[x]['title']) > 50 else f"{x+1}. {all_data[x]['title']}"
                )
                
                if selected_product is not None:
                    product = all_data[selected_product]
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**🔗 URL:** {product['url']}")
                        st.markdown(f"**📝 Título:** {product['title']}")
                        if product['price']:
                            st.markdown(f"**💰 Precio:** {product['price']}")
                        if product['description']:
                            st.markdown(f"**📄 Descripción:** {product['description'][:200]}...")
                    
                    with col2:
                        if product['features']:
                            st.markdown("**⭐ Características principales:**")
                            for feature in product['features'][:5]:
                                st.markdown(f"• {feature}")
        
        with tab2:
            if analyze_terms:
                st.header("🔤 Términos Más Repetidos")
                
                terms = analyzer.analyze_terms(all_data)
                top_terms = terms.most_common(top_n)
                
                if top_terms:
                    df_terms = pd.DataFrame(top_terms, columns=['Término', 'Frecuencia'])
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        fig = px.bar(
                            df_terms, 
                            x='Frecuencia', 
                            y='Término',
                            orientation='h', 
                            title="Términos más frecuentes",
                            color='Frecuencia',
                            color_continuous_scale='viridis'
                        )
                        fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("**📊 Top Términos:**")
                        st.dataframe(df_terms, use_container_width=True, hide_index=True)
                        
                        # Insights automáticos
                        st.markdown("**💡 Insights:**")
                        if len(top_terms) > 0:
                            most_common = top_terms[0]
                            st.info(f"El término más común es **'{most_common[0]}'** con {most_common[1]} apariciones")
                        
                        # Palabras técnicas vs comerciales
                        technical_words = [term for term, count in top_terms if any(tech in term.lower() for tech in ['tech', 'digital', 'smart', 'pro', 'hd', '4k', 'wifi', 'bluetooth'])]
                        if technical_words:
                            st.success(f"Se detectaron {len(technical_words)} términos técnicos en el top {top_n}")
        
        with tab3:
            if analyze_filters:
                st.header("🎛️ Filtros Más Usados")
                
                filters = analyzer.analyze_filters(all_data)
                top_filters = filters.most_common(top_n)
                
                if top_filters:
                    df_filters = pd.DataFrame(top_filters, columns=['Filtro', 'Frecuencia'])
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Gráfico de dona
                        fig = px.pie(
                            df_filters.head(15), 
                            values='Frecuencia', 
                            names='Filtro',
                            title="Distribución de Filtros Más Comunes",
                            hole=0.4
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("**🎛️ Top Filtros:**")
                        st.dataframe(df_filters, use_container_width=True, hide_index=True)
                        
                        # Insights sobre filtros
                        st.markdown("**💡 Insights:**")
                        if len(top_filters) > 0:
                            st.info(f"El filtro más común es **'{top_filters[0][0]}'**")
                        
                        # Categorizar tipos de filtros
                        filter_categories = {
                            'Precio': ['precio', 'price', 'cost', 'euro', 'dollar'],
                            'Marca': ['marca', 'brand', 'fabricante'],
                            'Color': ['color', 'colour', 'negro', 'blanco', 'rojo'],
                            'Tamaño': ['tamaño', 'size', 'talla', 'grande', 'pequeño']
                        }
                        
                        for category, keywords in filter_categories.items():
                            category_filters = [f for f, c in top_filters if any(kw in f.lower() for kw in keywords)]
                            if category_filters:
                                st.success(f"**{category}:** {len(category_filters)} filtros detectados")
        
        with tab4:
            if analyze_features:
                st.header("⭐ Características Más Mencionadas")
                
                features = analyzer.analyze_features(all_data)
                top_features = features.most_common(top_n)
                
                if top_features:
                    df_features = pd.DataFrame(top_features, columns=['Característica', 'Frecuencia'])
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Gráfico de burbujas
                        fig = px.scatter(
                            df_features, 
                            x='Frecuencia', 
                            y='Característica',
                            size='Frecuencia', 
                            title="Características más mencionadas",
                            color='Frecuencia',
                            color_continuous_scale='plasma',
                            size_max=30
                        )
                        fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("**⭐ Top Características:**")
                        st.dataframe(df_features, use_container_width=True, hide_index=True)
                        
                        # Análisis de características
                        st.markdown("**💡 Insights:**")
                        if len(top_features) > 0:
                            st.info(f"La característica más mencionada es **'{top_features[0][0]}'**")
                        
                        # Detectar categorías de características
                        tech_features = [f for f, c in top_features if any(tech in f.lower() for tech in ['digital', 'smart', 'auto', 'wireless', 'bluetooth', 'wifi'])]
                        comfort_features = [f for f, c in top_features if any(comfort in f.lower() for tech in ['comfort', 'ergonomic', 'soft', 'light'])]
                        
                        if tech_features:
                            st.success(f"**Tecnológicas:** {len(tech_features)} características")
                        if comfort_features:
                            st.success(f"**Comodidad:** {len(comfort_features)} características")
        
        with tab5:
            st.header("📈 Visualizaciones Adicionales")
            
            # Nube de palabras
            if show_wordcloud and analyze_terms and WORDCLOUD_AVAILABLE:
                st.subheader("☁️ Nube de Palabras")
                
                terms = analyzer.analyze_terms(all_data)
                if terms:
                    try:
                        # Configurar matplotlib para que funcione en Streamlit Cloud
                        plt.rcParams['font.family'] = ['DejaVu Sans']
                        
                        wordcloud = WordCloud(
                            width=1000, 
                            height=500,
                            background_color='white',
                            colormap='viridis',
                            max_words=100,
                            relative_scaling=0.5,
                            random_state=42
                        ).generate_from_frequencies(dict(terms.most_common(100)))
                        
                        fig, ax = plt.subplots(figsize=(15, 8))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)
                        plt.close()
                    except Exception as e:
                        st.error(f"Error generando nube de palabras: {e}")
                        st.info("Mostrando términos en formato de tabla como alternativa:")
                        terms_df = pd.DataFrame(terms.most_common(50), columns=['Término', 'Frecuencia'])
                        st.dataframe(terms_df)
            elif show_wordcloud and analyze_terms and not WORDCLOUD_AVAILABLE:
                st.subheader("📊 Términos Principales (WordCloud no disponible)")
                terms = analyzer.analyze_terms(all_data)
                if terms:
                    terms_df = pd.DataFrame(terms.most_common(50), columns=['Término', 'Frecuencia'])
                    
                    # Crear gráfico de barras como alternativa
                    fig = px.bar(
                        terms_df.head(20),
                        x='Frecuencia',
                        y='Término',
                        orientation='h',
                        title="Top 20 Términos Más Frecuentes",
                        color='Frecuencia',
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
            
            # Análisis comparativo entre productos
            st.subheader("📊 Análisis Comparativo")
            
            if len(all_data) > 1:
                # Matriz de características por producto
                st.markdown("**🎯 Matriz de Presencia de Características**")
                
                # Obtener todas las características únicas
                all_unique_features = set()
                for data in all_data:
                    feature_words = []
                    for feature in data.get('features', []):
                        words = re.findall(r'\b[a-záéíóúñüA-ZÁÉÍÓÚÑÜ]{3,}\b', feature.lower())
                        feature_words.extend([w for w in words if w not in analyzer.stop_words])
                    all_unique_features.update(feature_words)
                
                # Tomar solo las características más comunes
                feature_counts = Counter()
                for data in all_data:
                    feature_words = []
                    for feature in data.get('features', []):
                        words = re.findall(r'\b[a-záéíóúñüA-ZÁÉÍÓÚÑÜ]{3,}\b', feature.lower())
                        feature_words.extend([w for w in words if w not in analyzer.stop_words])
                    feature_counts.update(feature_words)
                
                top_feature_words = [f[0] for f in feature_counts.most_common(15)]
                
                if top_feature_words:
                    matrix_data = []
                    for i, data in enumerate(all_data):
                        row = {}
                        feature_words = []
                        for feature in data.get('features', []):
                            words = re.findall(r'\b[a-záéíóúñüA-ZÁÉÍÓÚÑÜ]{3,}\b', feature.lower())
                            feature_words.extend([w for w in words if w not in analyzer.stop_words])
                        
                        for feature_word in top_feature_words:
                            row[feature_word] = 1 if feature_word in feature_words else 0
                        
                        matrix_data.append(row)
                    
                    if matrix_data:
                        df_matrix = pd.DataFrame(matrix_data)
                        df_matrix.index = [f"Producto {i+1}" for i in range(len(df_matrix))]
                        
                        # Usar plotly para la matriz
                        fig = px.imshow(
                            df_matrix.values,
                            labels=dict(x="Características", y="Productos", color="Presente"),
                            x=df_matrix.columns,
                            y=df_matrix.index,
                            color_continuous_scale='RdYlBu_r',
                            title="Matriz de Características por Producto"
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Mostrar estadísticas de la matriz
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            unique_features_per_product = df_matrix.sum(axis=1).mean()
                            st.metric("📊 Promedio características/producto", f"{unique_features_per_product:.1f}")
                        with col2:
                            most_common_feature = df_matrix.sum(axis=0).idxmax()
                            st.metric("🏆 Característica más común", most_common_feature)
                        with col3:
                            coverage = (df_matrix.sum(axis=0) > 0).sum() / len(df_matrix.columns) * 100
                            st.metric("📈 Cobertura de características", f"{coverage:.1f}%")
            
            # Análisis de dominios
            st.subheader("🌐 Análisis por Dominio")
            
            domain_analysis = {}
            for data in all_data:
                domain = urlparse(data['url']).netloc
                if domain not in domain_analysis:
                    domain_analysis[domain] = {
                        'productos': 0,
                        'caracteristicas_total': 0,
                        'con_precio': 0,
                        'specs_total': 0
                    }
                
                domain_analysis[domain]['productos'] += 1
                domain_analysis[domain]['caracteristicas_total'] += len(data.get('features', []))
                domain_analysis[domain]['specs_total'] += len(data.get('specifications', {}))
                if data.get('price'):
                    domain_analysis[domain]['con_precio'] += 1
            
            if len(domain_analysis) > 1:
                domain_df = pd.DataFrame(domain_analysis).T
                domain_df['promedio_caracteristicas'] = domain_df['caracteristicas_total'] / domain_df['productos']
                domain_df['porcentaje_con_precio'] = (domain_df['con_precio'] / domain_df['productos']) * 100
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(
                        domain_df.reset_index(),
                        x='index',
                        y='promedio_caracteristicas',
                        title="Promedio de Características por Dominio",
                        labels={'index': 'Dominio', 'promedio_caracteristicas': 'Promedio Características'}
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.bar(
                        domain_df.reset_index(),
                        x='index',
                        y='porcentaje_con_precio',
                        title="Porcentaje de Productos con Precio por Dominio",
                        labels={'index': 'Dominio', 'porcentaje_con_precio': 'Porcentaje con Precio (%)'}
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(domain_df, use_container_width=True)
        
        # Sección de descarga y exportación
        st.markdown("---")
        st.header("💾 Exportar Resultados")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Generar Reporte CSV", use_container_width=True):
                # Crear reporte consolidado
                report_data = []
                
                if analyze_terms:
                    terms = analyzer.analyze_terms(all_data)
                    for term, count in terms.most_common(top_n):
                        report_data.append({
                            'Tipo': 'Término',
                            'Elemento': term,
                            'Frecuencia': count,
                            'Porcentaje': f"{(count/sum(terms.values())*100):.1f}%"
                        })
                
                if analyze_filters:
                    filters = analyzer.analyze_filters(all_data)
                    for filter_item, count in filters.most_common(top_n):
                        report_data.append({
                            'Tipo': 'Filtro',
                            'Elemento': filter_item,
                            'Frecuencia': count,
                            'Porcentaje': f"{(count/sum(filters.values())*100):.1f}%" if sum(filters.values()) > 0 else "0%"
                        })
                
                if analyze_features:
                    features = analyzer.analyze_features(all_data)
                    for feature, count in features.most_common(top_n):
                        report_data.append({
                            'Tipo': 'Característica',
                            'Elemento': feature,
                            'Frecuencia': count,
                            'Porcentaje': f"{(count/sum(features.values())*100):.1f}%" if sum(features.values()) > 0 else "0%"
                        })
                
                df_report = pd.DataFrame(report_data)
                csv = df_report.to_csv(index=False, encoding='utf-8')
                
                st.download_button(
                    label="📥 Descargar Análisis CSV",
                    data=csv,
                    file_name=f"analisis_competencia_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📋 Generar Reporte Productos", use_container_width=True):
                # Crear reporte de productos
                products_data = []
                for i, data in enumerate(all_data):
                    products_data.append({
                        'ID': i + 1,
                        'URL': data['url'],
                        'Dominio': urlparse(data['url']).netloc,
                        'Título': data['title'],
                        'Precio': data['price'],
                        'Descripción': data['description'][:200] + '...' if len(data['description']) > 200 else data['description'],
                        'Num_Características': len(data['features']),
                        'Num_Especificaciones': len(data['specifications']),
                        'Num_Filtros': len(data['filters']),
                        'Características': ' | '.join(data['features'][:5]),  # Primeras 5 características
                        'Fecha_Análisis': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                
                df_products = pd.DataFrame(products_data)
                csv_products = df_products.to_csv(index=False, encoding='utf-8')
                
                st.download_button(
                    label="📥 Descargar Productos CSV",
                    data=csv_products,
                    file_name=f"productos_analizados_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("📈 Generar Reporte Excel", use_container_width=True):
                # Para Excel necesitaríamos openpyxl, por ahora ofrecemos CSV mejorado
                st.info("💡 Para Excel, descarga los CSV y ábrelos en Excel. ¡Funciona perfectamente!")
        
        # Insights finales y recomendaciones
        st.markdown("---")
        st.header("💡 Insights y Recomendaciones")
        
        # Generar insights automáticos
        insights = []
        
        if analyze_terms:
            terms = analyzer.analyze_terms(all_data)
            if terms:
                most_common_term = terms.most_common(1)[0]
                insights.append(f"🔤 **Término más relevante**: '{most_common_term[0]}' aparece {most_common_term[1]} veces")
        
        if analyze_features:
            features = analyzer.analyze_features(all_data)
            if features:
                most_common_feature = features.most_common(1)[0]
                insights.append(f"⭐ **Característica clave**: '{most_common_feature[0]}' es mencionada {most_common_feature[1]} veces")
        
        # Análisis de precios
        prices_available = sum(1 for data in all_data if data.get('price'))
        if prices_available > 0:
            price_coverage = (prices_available / len(all_data)) * 100
            insights.append(f"💰 **Transparencia de precios**: {price_coverage:.1f}% de productos muestran precio")
        
        # Análisis de completitud
        avg_features = sum(len(data.get('features', [])) for data in all_data) / len(all_data)
        insights.append(f"📊 **Promedio de características**: {avg_features:.1f} por producto")
        
        # Mostrar insights
        for insight in insights:
            st.markdown(insight)
        
        # Recomendaciones basadas en el análisis
        st.subheader("🎯 Recomendaciones Estratégicas")
        
        recommendations = []
        
        if analyze_terms:
            terms = analyzer.analyze_terms(all_data)
            top_terms = [term for term, count in terms.most_common(10)]
            recommendations.append(f"**SEO y Contenido**: Incorpora estos términos clave en tus descripciones: {', '.join(top_terms[:5])}")
        
        if analyze_features:
            features = analyzer.analyze_features(all_data)
            top_features = [feature for feature, count in features.most_common(5)]
            recommendations.append(f"**Desarrollo de Producto**: Considera destacar estas características: {', '.join(top_features[:3])}")
        
        if analyze_filters:
            filters = analyzer.analyze_filters(all_data)
            if filters:
                recommendations.append(f"**UX/UI**: Implementa filtros similares a la competencia para mejorar la navegación")
        
        # Análisis de gaps
        if len(all_data) > 2:
            # Buscar características únicas (que aparecen en pocos productos)
            all_feature_words = []
            for data in all_data:
                for feature in data.get('features', []):
                    words = re.findall(r'\b[a-záéíóúñüA-ZÁÉÍÓÚÑÜ]{4,}\b', feature.lower())
                    all_feature_words.extend([w for w in words if w not in analyzer.stop_words])
            
            feature_counts = Counter(all_feature_words)
            rare_features = [feature for feature, count in feature_counts.items() if count == 1]
            
            if rare_features:
                recommendations.append(f"**Oportunidades**: Explora características únicas como: {', '.join(rare_features[:3])}")
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
        
        # Call to action final
        st.markdown("---")
        st.success("🎉 **¡Análisis completado!** Usa estos insights para optimizar tu estrategia de producto y marketing.")
        
        # Información adicional
        with st.expander("ℹ️ Cómo usar estos resultados"):
            st.markdown("""
            **Para Marketing:**
            - Usa los términos más frecuentes en tus campañas publicitarias
            - Incorpora las características más mencionadas en tu copy
            - Analiza los filtros para mejorar la categorización de productos
            
            **Para Desarrollo de Producto:**
            - Identifica gaps en características que puedes cubrir
            - Benchmarkea especificaciones técnicas
            - Detecta tendencias emergentes en features
            
            **Para UX/UI:**
            - Implementa filtros similares a la competencia
            - Mejora la estructura de información de productos
            - Optimiza la presentación de características
            
            **Para SEO:**
            - Incorpora términos clave en títulos y descripciones
            - Optimiza meta descriptions con palabras relevantes
            - Mejora la arquitectura de información del sitio
            """)

if __name__ == "__main__":
    main()
