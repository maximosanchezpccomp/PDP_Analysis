import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from collections import Counter
import re
from urllib.parse import urlparse
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
        try:
            nltk.download('punkt')
        except:
            pass
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        try:
            nltk.download('stopwords')
        except:
            pass

download_nltk_data()

class ProductBenchmarkAnalyzer:
    def __init__(self):
        try:
            # Stopwords básicas en español e inglés
            spanish_stopwords = set([
                'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 
                'le', 'da', 'su', 'por', 'son', 'con', 'para', 'al', 'del', 'las', 'una', 
                'su', 'me', 'si', 'tu', 'más', 'muy', 'pero', 'como', 'son', 'los', 'este',
                'esta', 'esto', 'ese', 'esa', 'esos', 'esas', 'tiene', 'ser', 'hacer',
                'estar', 'todo', 'todos', 'toda', 'todas', 'cuando', 'donde', 'como',
                'porque', 'aunque', 'desde', 'hasta', 'entre', 'sobre', 'bajo', 'sin'
            ])
            
            english_stopwords = set([
                'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
                'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
                'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 
                'might', 'must', 'can', 'this', 'that', 'these', 'those', 'all', 'any', 
                'some', 'each', 'every', 'both', 'either', 'neither', 'one', 'two', 'three'
            ])
            
            # Palabras relacionadas con e-commerce que NO queremos analizar
            ecommerce_stopwords = set([
                'añadir', 'carrito', 'comprar', 'compra', 'pedido', 'envio', 'envío', 
                'entrega', 'prevista', 'generado', 'stock', 'disponible', 'agotado',
                'precio', 'oferta', 'descuento', 'rebaja', 'promocion', 'promoción',
                'gratis', 'gratuito', 'iva', 'incluido', 'excluido', 'gastos',
                'valoracion', 'valoración', 'opinion', 'opinión', 'comentario',
                'puntuacion', 'puntuación', 'estrella', 'estrellas', 'valorar',
                'recomendar', 'recomiendo', 'cliente', 'clientes', 'usuario', 'usuarios',
                'cada', 'solo', 'sólo', 'solamente', 'únicamente', 'también', 'además',
                'producto', 'productos', 'articulo', 'artículo', 'item', 'items',
                'marca', 'modelo', 'referencia', 'codigo', 'código', 'sku',
                'categoria', 'categoría', 'seccion', 'sección', 'departamento',
                'buscar', 'busqueda', 'búsqueda', 'filtrar', 'filtro', 'filtros',
                'ordenar', 'clasificar', 'mostrar', 'ver', 'todos', 'todas',
                'inicio', 'home', 'tienda', 'shop', 'store', 'online',
                'web', 'website', 'pagina', 'página', 'sitio', 'portal',
                'cookies', 'politica', 'política', 'privacidad', 'terminos', 'términos',
                'condiciones', 'legal', 'aviso', 'contacto', 'ayuda', 'soporte'
            ])
            
            try:
                nltk_spanish = set(nltk.corpus.stopwords.words('spanish'))
                nltk_english = set(nltk.corpus.stopwords.words('english'))
                self.stop_words = spanish_stopwords | english_stopwords | ecommerce_stopwords | nltk_spanish | nltk_english
            except:
                self.stop_words = spanish_stopwords | english_stopwords | ecommerce_stopwords
                
        except:
            # Fallback mínimo
            self.stop_words = set(['el', 'la', 'de', 'que', 'y', 'a', 'en', 'the', 'and', 'or', 'añadir', 'carrito', 'entrega', 'envio'])
        
        self.results = []
        
    def extract_content_from_url(self, url):
        """Extrae contenido relevante de una URL de producto"""
        try:
            # Headers más sofisticados para evitar detección de bot
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                'Accept-Language': 'es-ES,es;q=0.9,en;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Cache-Control': 'max-age=0'
            }
            
            # Usar session para mantener cookies
            session = requests.Session()
            session.headers.update(headers)
            
            response = session.get(url, timeout=20, allow_redirects=True)
            
            # Si obtenemos 403, intentamos con diferentes estrategias
            if response.status_code == 403:
                # Estrategia alternativa
                alternative_headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'es-ES,es;q=0.5'
                }
                session.headers.clear()
                session.headers.update(alternative_headers)
                time.sleep(2)
                response = session.get(url, timeout=20, allow_redirects=True)
            
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
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                st.warning(f"🚫 Acceso denegado a {urlparse(url).netloc} - El sitio bloquea bots automáticos")
            else:
                st.warning(f"⚠️ Error HTTP {e.response.status_code} con {url[:50]}...")
            return None
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
        """Extrae la descripción del producto enfocándose en contenido relevante"""
        description = ""
        
        # Selectores específicos para descripciones de producto
        description_selectors = [
            '[class*="product-description"]',
            '[class*="description"]',
            '[class*="summary"]',
            '[class*="overview"]',
            '[class*="details"]',
            '[data-testid*="description"]',
            '[class*="product-info"]',
            '[class*="caracteristicas"]',
            'meta[name="description"]'
        ]
        
        # Elementos a excluir
        excluded_classes = [
            'nav', 'menu', 'header', 'footer', 'sidebar', 'cart', 'carrito',
            'checkout', 'payment', 'shipping', 'delivery', 'price', 'precio',
            'review', 'opinion', 'rating', 'valoracion', 'breadcrumb'
        ]
        
        for selector in description_selectors:
            if 'meta' in selector:
                element = soup.select_one(selector)
                if element:
                    desc = element.get('content', '')
                    if desc and len(desc) > 30:
                        description += desc + " "
            else:
                elements = soup.select(selector)
                for element in elements:
                    # Verificar que no sea un elemento excluido
                    element_class = element.get('class', [])
                    element_id = element.get('id', '')
                    
                    is_excluded = any(
                        excluded in str(element_class).lower() or 
                        excluded in element_id.lower() 
                        for excluded in excluded_classes
                    )
                    
                    if not is_excluded:
                        text = element.get_text().strip()
                        if text and len(text) > 30 and len(text) < 3000:
                            if not self._is_ecommerce_text(text):
                                description += text + " "
        
        return description.strip()
    
    def _is_ecommerce_text(self, text):
        """Detecta si un texto es relacionado con e-commerce y no con producto"""
        text_lower = text.lower()
        
        # Patrones que indican texto de e-commerce
        ecommerce_patterns = [
            'añadir al carrito', 'comprar ahora', 'envío gratis',
            'opiniones de', 'valoraciones de', 'política de',
            'mi cuenta', 'iniciar sesión', 'comparar producto',
            'stock disponible', 'descuento del', 'gastos de envío'
        ]
        
        pattern_count = sum(1 for pattern in ecommerce_patterns if pattern in text_lower)
        
        # Si más del 30% del texto son palabras de e-commerce, lo descartamos
        words = text_lower.split()
        ecommerce_word_count = sum(1 for word in words if word in self.stop_words)
        ecommerce_ratio = ecommerce_word_count / len(words) if words else 0
        
        return pattern_count > 2 or ecommerce_ratio > 0.3
    
    def _extract_features(self, soup):
        """Extrae características y features del producto"""
        features = []
        
        # Buscar listas de características
        feature_selectors = [
            '[class*="feature"] li',
            '[class*="benefit"] li',
            '[class*="highlight"] li',
            '[class*="spec"] li',
            'ul[class*="feature"] li',
            '.features li',
            '.benefits li',
            'div[class*="feature"]'
        ]
        
        for selector in feature_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text().strip()
                if (text and 
                    len(text) > 10 and 
                    len(text) < 500 and 
                    not re.match(r'^\d+$', text) and
                    not text.lower().startswith(('http', 'www', 'mailto'))):
                    features.append(text)
        
        # Eliminar duplicados
        seen = set()
        unique_features = []
        for feature in features:
            if feature.lower() not in seen:
                seen.add(feature.lower())
                unique_features.append(feature)
        
        return unique_features[:50]
    
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
        
        return specs
    
    def _extract_price(self, soup):
        """Extrae información de precio"""
        price_selectors = [
            '[class*="price"]',
            '[class*="cost"]',
            '[class*="amount"]',
            '[data-testid*="price"]',
            '[id*="price"]'
        ]
        
        for selector in price_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text().strip()
                # Buscar patrones de precio
                price_patterns = [
                    r'[€$£¥]\s*[\d,]+\.?\d*',
                    r'[\d,]+\.?\d*\s*[€$£¥]',
                    r'[\d,]+\.?\d*\s*EUR?'
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
            'select option',
            '[type="checkbox"] + label'
        ]
        
        for selector in filter_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text().strip()
                if (text and 
                    len(text) > 2 and 
                    len(text) < 80 and
                    not text.lower().startswith(('http', 'www')) and
                    not re.match(r'^\d+$', text)):
                    filters.append(text)
        
        return list(set(filters))[:100]
    
    def _extract_categories(self, soup):
        """Extrae categorías del producto"""
        categories = []
        
        category_selectors = [
            '[class*="breadcrumb"] a',
            '[class*="category"] a',
            '.breadcrumb a'
        ]
        
        for selector in category_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text().strip()
                if (text and 
                    text.lower() not in ['home', 'inicio', 'tienda'] and
                    len(text) > 2 and 
                    len(text) < 50):
                    categories.append(text)
        
        return categories
    
    def analyze_terms(self, all_data):
        """Analiza los términos más frecuentes enfocándose en características de producto"""
        all_text = ""
        
        for data in all_data:
            # Priorizar título y características
            title_text = data.get('title', '')
            features_text = " ".join(data.get('features', []))
            specs_keys = " ".join(data.get('specifications', {}).keys())
            specs_values = " ".join(data.get('specifications', {}).values())
            
            # Dar más peso a características técnicas
            all_text += f" {title_text} {features_text} {features_text} {specs_keys} {specs_values} "
            
            # Agregar descripción filtrada
            description = data.get('description', '')
            if description:
                sentences = description.split('.')
                for sentence in sentences:
                    if self._is_product_relevant_sentence(sentence):
                        all_text += sentence + " "
        
        # Limpiar y tokenizar texto
        words = re.findall(r'\b[a-záéíóúñüA-ZÁÉÍÓÚÑÜ]{3,}\b', all_text.lower())
        
        # Filtrar palabras relevantes
        filtered_words = []
        for word in words:
            if (word not in self.stop_words and 
                len(word) >= 3 and 
                not word.isdigit() and
                self._is_product_term(word)):
                filtered_words.append(word)
        
        return Counter(filtered_words)
    
    def _is_product_relevant_sentence(self, sentence):
        """Determina si una oración es relevante para el producto"""
        sentence_lower = sentence.lower().strip()
        
        # Frases que indican características técnicas
        positive_indicators = [
            'características', 'especificaciones', 'incluye', 'cuenta con',
            'tecnología', 'material', 'diseño', 'tamaño', 'dimensiones',
            'memoria', 'procesador', 'pantalla', 'batería', 'compatible'
        ]
        
        # Frases no relevantes
        negative_indicators = [
            'añadir', 'carrito', 'comprar', 'precio', 'envío',
            'opinión', 'valoración', 'stock', 'oferta', 'cliente'
        ]
        
        positive_score = sum(1 for indicator in positive_indicators if indicator in sentence_lower)
        negative_score = sum(1 for indicator in negative_indicators if indicator in sentence_lower)
        
        return positive_score > negative_score and len(sentence.strip()) > 20
    
    def _is_product_term(self, word):
        """Determina si una palabra es relevante para describir productos"""
        irrelevant_terms = {
            'página', 'sitio', 'web', 'usuario', 'cliente', 'cuenta',
            'compra', 'pedido', 'pago', 'envío', 'precio', 'oferta',
            'opinión', 'valoración', 'comentario', 'estrella'
        }
        
        return word not in irrelevant_terms
    
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
    # CSS personalizado
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 0.75rem 1.25rem;
        margin-bottom: 1rem;
        color: #155724;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header principal
    st.markdown('<h1 class="main-header">📊 Herramienta de Análisis de Competencia</h1>', unsafe_allow_html=True)
    st.markdown("### Analiza fichas de productos de la competencia para obtener insights clave")
    
    # Mensaje de estado de librerías
    if not WORDCLOUD_AVAILABLE:
        st.info("ℹ️ WordCloud no está disponible. Las nubes de palabras se mostrarán como gráficos de barras.")
    
    # Información de ayuda
    with st.expander("ℹ️ ¿Cómo funciona esta herramienta?"):
        st.markdown("""
        **Esta herramienta te permite:**
        
        1. 🔗 **Analizar múltiples URLs de productos** de diferentes sitios web
        2. 📊 **Extraer automáticamente** títulos, descripciones, características y precios
        3. 🔍 **Identificar patrones** en términos, filtros y features más comunes
        4. 📈 **Visualizar los resultados** con gráficos interactivos
        5. 💾 **Exportar los datos** en formato CSV para análisis adicionales
        
        **✅ Sitios web compatibles:**
        - **Amazon** (amazon.es, amazon.com) - ⭐ Recomendado
        - **eBay** (ebay.es, ebay.com) - ⭐ Recomendado  
        - **AliExpress** - Generalmente funciona bien
        - **Tiendas online pequeñas** - Menos restrictivas
        
        **🚫 Sitios con restricciones:**
        - **MediaMarkt, PCComponentes, El Corte Inglés** - Requieren modo agresivo
        - **Grandes retailers** - Pueden bloquear bots automáticos
        
        **💡 Consejos:**
        - Usa URLs de productos específicos (no categorías)
        - Activa el "Modo agresivo" para sitios problemáticos
        - Aumenta el delay a 3-5 segundos para evitar bloqueos
        """)
        
    # Aviso sobre términos filtrados
    st.info("🎯 **Análisis optimizado:** La herramienta filtra automáticamente términos relacionados con compra, envío, opiniones, etc. para centrarse en características reales del producto.")
    
    # Sidebar para configuración
    st.sidebar.header("⚙️ Configuración del Análisis")
    st.sidebar.markdown("---")
    
    # Opciones de análisis
    st.sidebar.subheader("📋 Tipos de Análisis")
    analyze_terms = st.sidebar.checkbox("🔤 Términos más repetidos", value=True)
    analyze_filters = st.sidebar.checkbox("🎛️ Filtros más usados", value=True)
    analyze_features = st.sidebar.checkbox("⭐ Características más mencionadas", value=True)
    
    # Solo mostrar opción de wordcloud si está disponible
    if WORDCLOUD_AVAILABLE:
        show_wordcloud = st.sidebar.checkbox("☁️ Nube de palabras", value=True)
    else:
        show_wordcloud = False
        st.sidebar.info("☁️ Nube de palabras no disponible")
    
    st.sidebar.markdown("---")
    
    # Configuración de límites
    st.sidebar.subheader("🎯 Configuración de Resultados")
    top_n = st.sidebar.slider("📊 Top N resultados", 5, 50, 20)
    
    # Configuración de scraping
    st.sidebar.subheader("🔧 Configuración Avanzada")
    delay = st.sidebar.slider("⏱️ Delay entre requests (seg)", 0.5, 5.0, 2.0, 0.5)
    
    # Opciones para sitios problemáticos
    st.sidebar.markdown("**🛡️ Anti-detección:**")
    retry_403 = st.sidebar.checkbox("🔄 Reintentar URLs bloqueadas", value=True)
    
    aggressive_mode = st.sidebar.checkbox("🚀 Modo agresivo", value=False,
                                        help="Delays más largos y más reintentos")
    
    if aggressive_mode:
        delay = max(delay, 3.0)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("💡 **Tips:**")
    st.sidebar.markdown("- Activa modo agresivo para sitios problemáticos")
    st.sidebar.markdown("- Usa delays de 3-5 seg para evitar bloqueos")
    
    # Input de URLs
    st.header("🔗 URLs de Productos a Analizar")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        urls_input = st.text_area(
            "Introduce las URLs de productos (una por línea):",
            height=200,
            placeholder="""https://www.amazon.es/dp/B08N5WRWNW
https://www.amazon.es/dp/B087DTHJ8B
https://www.amazon.es/dp/B08CH7RHDP""",
            help="Pega aquí las URLs completas de los productos que quieres analizar"
        )
    
    with col2:
        st.markdown("**📝 URLs que funcionan bien:**")
        st.code("amazon.es/dp/B08X...")
        st.code("ebay.es/itm/...")
        st.code("aliexpress.com/item/...")
        st.markdown("**🚫 URLs problemáticas:**")
        st.markdown("- mediamarkt.es")
        st.markdown("- pccomponentes.com")
        st.markdown("- elcorteingles.es")
        st.markdown("**💡 Tip:** Usa el modo agresivo para sitios problemáticos")
    
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
        
        if valid_urls:
            st.success(f"✅ {len(valid_urls)} URLs válidas detectadas")
    
    # Botón principal de análisis
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🚀 INICIAR ANÁLISIS COMPLETO", 
                                 type="primary", 
                                 use_container_width=True)
    
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
            
            # Aplicar delay más largo si es modo agresivo
            if i > 0:
                current_delay = delay * 1.5 if aggressive_mode else delay
                time.sleep(current_delay)
            
            data = analyzer.extract_content_from_url(url)
            if data:
                all_data.append(data)
                success_metric.metric("✅ Exitosos", len(all_data))
            else:
                failed_count += 1
                failed_metric.metric("❌ Fallidos", failed_count)
                
                # Si está activado el retry y falló, intentar una vez más
                if retry_403 and failed_count <= 3:
                    status_text.markdown(f'🔄 **Reintentando URL bloqueada...**')
                    time.sleep(5)
                    retry_data = analyzer.extract_content_from_url(url)
                    if retry_data:
                        all_data.append(retry_data)
                        success_metric.metric("✅ Exitosos", len(all_data))
                        failed_count -= 1
                        failed_metric.metric("❌ Fallidos", failed_count)
            
            progress_bar.progress((i + 1) / len(urls))
        
        status_text.markdown('✅ **Análisis completado exitosamente**')
        
        if not all_data:
            st.error("❌ No se pudo extraer información de ninguna URL.")
            
            st.info("""
            💡 **Sitios con protección anti-bot detectados:**
            
            **MediaMarkt, PCComponentes, El Corte Inglés** y otros grandes retailers
            suelen bloquear scraping automático por políticas de seguridad.
            
            **Alternativas recomendadas:**
            - Usa URLs de Amazon (menos restrictivo)
            - Prueba con tiendas online más pequeñas
            - Activa el "Modo agresivo" en configuración avanzada
            - Aumenta el delay entre requests a 3-5 segundos
            
            **URLs que suelen funcionar mejor:**
            - amazon.es, amazon.com
            - Tiendas especializadas más pequeñas
            - Sitios web de fabricantes
            - Marketplaces menos restrictivos
            """)
            return
        
        # Mostrar mensaje de éxito
        success_msg = f"""
        <div class="success-message">
            <strong>🎉 ¡Análisis completado!</strong><br>
            Se procesaron <strong>{len(all_data)}</strong> de <strong>{len(urls)}</strong> productos exitosamente
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
                        
                        # Insights mejorados
                        st.markdown("**💡 Insights:**")
                        if len(top_terms) > 0:
                            most_common = top_terms[0]
                            st.info(f"Término más relevante: **'{most_common[0]}'** ({most_common[1]} veces)")
                        
                        # Categorización automática de términos
                        tech_terms = [term for term, count in top_terms if any(tech in term.lower() for tech in ['digital', 'smart', 'tech', 'pro', 'hd', '4k', 'wifi', 'bluetooth', 'usb', 'led'])]
                        material_terms = [term for term, count in top_terms if any(mat in term.lower() for mat in ['acero', 'metal', 'plastico', 'madera', 'cristal', 'ceramic', 'silicon'])]
                        size_terms = [term for term, count in top_terms if any(size in term.lower() for size in ['grande', 'pequeño', 'mini', 'xl', 'slim', 'compact'])]
                        
                        if tech_terms:
                            st.success(f"**Tecnológicos:** {len(tech_terms)} términos detectados")
                        if material_terms:
                            st.success(f"**Materiales:** {len(material_terms)} términos detectados")
                        if size_terms:
                            st.success(f"**Tamaños:** {len(size_terms)} términos detectados")
        
        with tab3:
            if analyze_filters:
                st.header("🎛️ Filtros Más Usados")
                
                filters = analyzer.analyze_filters(all_data)
                top_filters = filters.most_common(top_n)
                
                if top_filters:
                    df_filters = pd.DataFrame(top_filters, columns=['Filtro', 'Frecuencia'])
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        fig = px.pie(
                            df_filters.head(15), 
                            values='Frecuencia', 
                            names='Filtro',
                            title="Distribución de Filtros Más Comunes"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("**🎛️ Top Filtros:**")
                        st.dataframe(df_filters, use_container_width=True, hide_index=True)
                        
                        # Insights sobre filtros mejorados
                        st.markdown("**💡 Insights:**")
                        if len(top_filters) > 0:
                            st.info(f"Filtro más común: **'{top_filters[0][0]}'**")
                        
                        # Categorización mejorada de filtros
                        filter_categories = {
                            'Precio': ['precio', 'price', 'cost', 'euro', 'dollar', 'barato', 'caro'],
                            'Marca': ['marca', 'brand', 'fabricante', 'sony', 'samsung', 'apple', 'lg'],
                            'Color': ['color', 'colour', 'negro', 'blanco', 'rojo', 'azul', 'verde'],
                            'Tamaño': ['tamaño', 'size', 'talla', 'grande', 'pequeño', 'xl', 'l', 'm', 's'],
                            'Características': ['wifi', 'bluetooth', 'smart', 'digital', 'hd', '4k', 'led']
                        }
                        
                        for category, keywords in filter_categories.items():
                            category_filters = [f for f, c in top_filters if any(kw in f.lower() for kw in keywords)]
                            if category_filters:
                                st.success(f"**{category}:** {len(category_filters)} filtros")
        
        with tab4:
            if analyze_features:
                st.header("⭐ Características Más Mencionadas")
                
                features = analyzer.analyze_features(all_data)
                top_features = features.most_common(top_n)
                
                if top_features:
                    df_features = pd.DataFrame(top_features, columns=['Característica', 'Frecuencia'])
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        fig = px.scatter(
                            df_features, 
                            x='Frecuencia', 
                            y='Característica',
                            size='Frecuencia', 
                            title="Características más mencionadas",
                            color='Frecuencia',
                            color_continuous_scale='plasma'
                        )
                        fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("**⭐ Top Características:**")
                        st.dataframe(df_features, use_container_width=True, hide_index=True)
                        
                        # Análisis de características mejorado
                        st.markdown("**💡 Insights:**")
                        if len(top_features) > 0:
                            st.info(f"Característica más mencionada: **'{top_features[0][0]}'**")
                        
                        # Detectar tipos de características
                        feature_types = {
                            'Tecnológicas': ['digital', 'smart', 'inteligente', 'automático', 'wireless', 'bluetooth', 'wifi', 'usb'],
                            'Diseño': ['elegante', 'moderno', 'compacto', 'ligero', 'resistente', 'duradero', 'premium'],
                            'Funcionalidad': ['fácil', 'práctico', 'versátil', 'multifuncional', 'eficiente', 'rápido', 'potente'],
                            'Comodidad': ['cómodo', 'ergonómico', 'suave', 'ajustable', 'flexible', 'antideslizante']
                        }
                        
                        for feature_type, keywords in feature_types.items():
                            type_features = [f for f, c in top_features if any(kw in f.lower() for kw in keywords)]
                            if type_features:
                                st.success(f"**{feature_type}:** {len(type_features)} características")
        
        with tab5:
            st.header("📈 Visualizaciones Adicionales")
            
            # Nube de palabras o alternativa
            if show_wordcloud and analyze_terms and WORDCLOUD_AVAILABLE:
                st.subheader("☁️ Nube de Palabras")
                
                terms = analyzer.analyze_terms(all_data)
                if terms:
                    try:
                        wordcloud = WordCloud(
                            width=1000, 
                            height=500,
                            background_color='white',
                            colormap='viridis',
                            max_words=100
                        ).generate_from_frequencies(dict(terms.most_common(100)))
                        
                        fig, ax = plt.subplots(figsize=(15, 8))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)
                        plt.close()
                    except Exception as e:
                        st.error(f"Error generando nube de palabras: {e}")
            elif analyze_terms:
                st.subheader("📊 Términos Principales")
                terms = analyzer.analyze_terms(all_data)
                if terms:
                    terms_df = pd.DataFrame(terms.most_common(20), columns=['Término', 'Frecuencia'])
                    
                    fig = px.bar(
                        terms_df,
                        x='Frecuencia',
                        y='Término',
                        orientation='h',
                        title="Top 20 Términos Más Frecuentes",
                        color='Frecuencia',
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
        
        # Sección de descarga
        st.markdown("---")
        st.header("💾 Exportar Resultados")
        
        if st.button("📊 Generar y Descargar Reporte CSV", use_container_width=True):
            # Crear reporte consolidado
            report_data = []
            
            if analyze_terms:
                terms = analyzer.analyze_terms(all_data)
                for term, count in terms.most_common(top_n):
                    report_data.append({
                        'Tipo': 'Término',
                        'Elemento': term,
                        'Frecuencia': count
                    })
            
            if analyze_filters:
                filters = analyzer.analyze_filters(all_data)
                for filter_item, count in filters.most_common(top_n):
                    report_data.append({
                        'Tipo': 'Filtro',
                        'Elemento': filter_item,
                        'Frecuencia': count
                    })
            
            if analyze_features:
                features = analyzer.analyze_features(all_data)
                for feature, count in features.most_common(top_n):
                    report_data.append({
                        'Tipo': 'Característica',
                        'Elemento': feature,
                        'Frecuencia': count
                    })
            
            df_report = pd.DataFrame(report_data)
            csv = df_report.to_csv(index=False, encoding='utf-8')
            
            st.download_button(
                label="📥 Descargar Análisis CSV",
                data=csv,
                file_name=f"analisis_competencia_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # Insights finales
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
            recommendations.append(f"**SEO y Contenido**: Incorpora estos términos clave: {', '.join(top_terms[:5])}")
        
        if analyze_features:
            features = analyzer.analyze_features(all_data)
            top_features = [feature for feature, count in features.most_common(5)]
            recommendations.append(f"**Desarrollo de Producto**: Destaca estas características: {', '.join(top_features[:3])}")
        
        if analyze_filters:
            filters = analyzer.analyze_filters(all_data)
            if filters:
                recommendations.append("**UX/UI**: Implementa filtros similares a la competencia")
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
        
        # Call to action final
        st.markdown("---")
        st.success("🎉 **¡Análisis completado!** Usa estos insights para optimizar tu estrategia de producto y marketing.")

if __name__ == "__main__":
    main()
