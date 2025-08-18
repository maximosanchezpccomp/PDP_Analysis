import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from collections import Counter
import re
from urllib.parse import urlparse, quote_plus
import time
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings
import json
import random

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
            nltk.download('punkt', quiet=True)
        except:
            pass
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        try:
            nltk.download('stopwords', quiet=True)
        except:
            pass

download_nltk_data()

class ProductBenchmarkAnalyzer:
    def __init__(self):
        try:
            # Stopwords básicas en español e inglés
            spanish_stopwords = set([
                'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se',
                'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para',
                'al', 'del', 'las', 'una', 'su', 'me', 'si', 'tu', 'más', 'muy',
                'pero', 'como', 'son', 'los', 'este', 'esta', 'esto', 'ese', 'esa',
                'esos', 'esas', 'tiene', 'ser', 'hacer', 'estar', 'todo', 'todos',
                'toda', 'todas', 'cuando', 'donde', 'como', 'porque', 'aunque',
                'desde', 'hasta', 'entre', 'sobre', 'bajo', 'sin'
            ])
            
            english_stopwords = set([
                'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
                'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
                'those', 'all', 'any', 'some', 'each', 'every', 'both', 'either',
                'neither', 'one', 'two', 'three'
            ])
            
            # Palabras relacionadas con e-commerce que NO queremos analizar
            ecommerce_stopwords = set([
                'añadir', 'carrito', 'comprar', 'compra', 'pedido', 'envio', 'envío',
                'entrega', 'prevista', 'generado', 'stock', 'disponible', 'agotado',
                'precio', 'oferta', 'descuento', 'rebaja', 'promocion', 'promoción',
                'gratis', 'gratuito', 'iva', 'incluido', 'excluido', 'gastos',
                'valoracion', 'valoración', 'opinion', 'opinión', 'comentario',
                'puntuacion', 'puntuación', 'estrella', 'estrellas', 'valorar',
                'recomendar', 'recomiendo', 'cliente', 'clientes', 'usuario',
                'usuarios', 'cada', 'solo', 'sólo', 'solamente', 'únicamente',
                'también', 'además', 'producto', 'productos', 'articulo', 'artículo',
                'item', 'items', 'marca', 'modelo', 'referencia', 'codigo', 'código',
                'sku', 'categoria', 'categoría', 'seccion', 'sección', 'departamento',
                'buscar', 'busqueda', 'búsqueda', 'filtrar', 'filtro', 'filtros',
                'ordenar', 'clasificar', 'mostrar', 'ver', 'todos', 'todas', 'inicio',
                'home', 'tienda', 'shop', 'store', 'online', 'web', 'website',
                'pagina', 'página', 'sitio', 'portal', 'cookies', 'politica',
                'política', 'privacidad', 'terminos', 'términos', 'condiciones',
                'legal', 'aviso', 'contacto', 'ayuda', 'soporte'
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
    
    def extract_content_from_url(self, url, rotate_headers=False):
        """Extrae contenido relevante de una URL de producto"""
        try:
            # Headers base más sofisticados
            headers_options = [
                {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                    'Accept-Language': 'es-ES,es;q=0.9,en;q=0.8',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                },
                {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'es-es',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Connection': 'keep-alive'
                },
                {
                    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'es,en-US;q=0.7,en;q=0.3',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive'
                }
            ]
            
            # Seleccionar headers
            if rotate_headers:
                headers = random.choice(headers_options)
            else:
                headers = headers_options[0]
            
            # Usar session para mantener cookies
            session = requests.Session()
            session.headers.update(headers)
            
            response = session.get(url, timeout=20, allow_redirects=True)
            
            # Si obtenemos 403, intentamos estrategias adicionales
            if response.status_code == 403:
                # Estrategia 1: Headers mínimos
                minimal_headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0'
                }
                session.headers.clear()
                session.headers.update(minimal_headers)
                time.sleep(3)
                response = session.get(url, timeout=20, allow_redirects=True)
                
                # Estrategia 2: Si sigue fallando, probar con otro user-agent
                if response.status_code == 403:
                    session.headers.update({
                        'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1'
                    })
                    time.sleep(5)
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
                domain = urlparse(url).netloc
                st.warning(f"🚫 Acceso denegado a {domain}")
                # Sugerir alternativas específicas
                if 'mediamarkt' in domain.lower():
                    st.info("💡 **Alternativa para MediaMarkt:** Busca el mismo producto en Amazon o eBay")
                elif 'pccomponentes' in domain.lower():
                    st.info("💡 **Alternativa para PCComponentes:** Prueba con Amazon o tiendas especializadas")
                elif 'elcorteingles' in domain.lower():
                    st.info("💡 **Alternativa para El Corte Inglés:** Busca en Amazon o tiendas del fabricante")
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
                        excluded in str(element_class).lower() or excluded in str(element_id).lower()
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
            'añadir al carrito', 'comprar ahora', 'envío gratis', 'opiniones de',
            'valoraciones de', 'política de', 'mi cuenta', 'iniciar sesión',
            'comparar producto', 'stock disponible', 'descuento del', 'gastos de envío'
        ]
        
        pattern_count = sum(1 for pattern in ecommerce_patterns if pattern in text_lower)
        
        # Si más del 30% del texto son palabras de e-commerce, lo descartamos
        words = text_lower.split()
        if not words:
            return False
            
        ecommerce_word_count = sum(1 for word in words if word in self.stop_words)
        ecommerce_ratio = ecommerce_word_count / len(words)
        
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
                if (text and len(text) > 10 and len(text) < 500 and
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
                if (text and len(text) > 2 and len(text) < 80 and
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
                if (text and text.lower() not in ['home', 'inicio', 'tienda'] and
                    len(text) > 2 and len(text) < 50):
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
            if (word not in self.stop_words and len(word) >= 3 and
                not word.isdigit() and self._is_product_term(word)):
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
            'añadir', 'carrito', 'comprar', 'precio', 'envío', 'opinión',
            'valoración', 'stock', 'oferta', 'cliente'
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

class GoogleShoppingAnalyzer:
    def __init__(self):
        self.serpapi_base = "https://serpapi.com/search"
        self.google_search_base = "https://www.googleapis.com/customsearch/v1"
    
    def search_products_free(self, query, num_results=20):
        """Método gratuito usando requests a Google Shopping"""
        try:
            search_url = f"https://www.google.com/search?tbm=shop&q={quote_plus(query)}&hl=es&gl=es"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'es-ES,es;q=0.9,en;q=0.8'
            }
            
            response = requests.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            products = []
            
            # Buscar diferentes tipos de contenedores de productos
            selectors = [
                'div[data-docid]',
                'div.sh-dlr__list-result',
                'div.mnr-c',
                'div.g'
            ]
            
            for selector in selectors:
                containers = soup.select(selector)
                for container in containers[:num_results]:
                    try:
                        # Extraer título
                        title_elem = (container.select_one('h3') or
                                    container.select_one('a[data-testid]') or
                                    container.select_one('a'))
                        title = title_elem.get_text().strip() if title_elem else ''
                        
                        # Extraer precio
                        price_elem = (container.select_one('span.a8Pemb') or
                                    container.select_one('[class*="price"]') or
                                    container.select_one('span:contains("€")'))
                        price = price_elem.get_text().strip() if price_elem else ''
                        
                        # Extraer tienda
                        source_elem = (container.select_one('.aULzUe') or
                                     container.select_one('[class*="merchant"]') or
                                     container.select_one('.F9PbJe'))
                        source = source_elem.get_text().strip() if source_elem else ''
                        
                        # Extraer enlace
                        link_elem = container.select_one('a[href]')
                        link = link_elem.get('href', '') if link_elem else ''
                        
                        if title and len(title) > 10:  # Solo productos con título válido
                            product = {
                                'title': title,
                                'price': price,
                                'source': source,
                                'link': link,
                                'description': title,
                                'method': 'Google Shopping Free'
                            }
                            products.append(product)
                    except Exception:
                        continue
                
                if len(products) >= num_results:
                    break
            
            return products[:num_results]
            
        except Exception as e:
            st.error(f"Error con búsqueda gratuita: {e}")
            return []
    
    def analyze_shopping_data(self, products):
        """Analiza los datos obtenidos de Google Shopping"""
        if not products:
            return {}
        
        analysis = {
            'total_products': len(products),
            'sources': {},
            'price_ranges': {},
            'common_terms': Counter()
        }
        
        # Análisis por fuente
        for product in products:
            source = product.get('source', 'Unknown').strip()
            if source:
                analysis['sources'][source] = analysis['sources'].get(source, 0) + 1
        
        # Análisis de precios
        prices = []
        for product in products:
            price_text = product.get('price', '')
            if price_text:
                price_match = re.search(r'[\d,]+\.?\d*', price_text.replace(',', ''))
                if price_match:
                    try:
                        price = float(price_match.group())
                        if price > 0 and price < 50000:  # Filtrar precios razonables
                            prices.append(price)
                    except:
                        pass
        
        if prices:
            analysis['price_ranges'] = {
                'min': min(prices),
                'max': max(prices),
                'avg': sum(prices) / len(prices),
                'count': len(prices)
            }
        
        # Análisis de términos comunes
        all_text = ''
        for product in products:
            all_text += f" {product.get('title', '')} {product.get('description', '')} "
        
        words = re.findall(r'\b[a-záéíóúñüA-ZÁÉÍÓÚÑÜ]{3,}\b', all_text.lower())
        
        stopwords = {'para', 'con', 'por', 'del', 'las', 'los', 'una', 'uno',
                    'the', 'and', 'for', 'with', 'desde', 'hasta', 'más', 'muy',
                    'todo', 'todos'}
        
        words = [word for word in words if word not in stopwords]
        analysis['common_terms'] = Counter(words)
        
        return analysis

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
