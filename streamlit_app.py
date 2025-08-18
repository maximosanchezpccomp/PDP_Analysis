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
                    not re.match(r'^\d+$', text) and
                    not text.lower().startswith(('http', 'www', 'mailto'))):
                    features.append(text)
        
        # Eliminar duplicados manteniendo orden
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
                    r'[\d,]+\.?\d*\s*EUR?',
                    r'[\d,]+\.?\d*\s*USD?'
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
                    text.lower() not in ['home', 'inicio', 'tienda', 'shop'] and
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
    delay = st.sidebar.slider("⏱️ Delay entre requests (seg)", 0.5, 5.0, 1.0, 0.5)
    
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
https://www.mediamarkt.es/producto-ejemplo-3""",
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
            
            data = analyzer.extract_content_from_url(url)
            if data:
                all_data.append(data)
                success_metric.metric("✅ Exitosos", len(all_data))
            else:
                failed_count += 1
                failed_metric.metric("❌ Fallidos", failed_count)
            
            progress_bar.progress((i + 1) / len(urls))
            time.sleep(delay)
        
        status_text.markdown('✅ **Análisis completado exitosamente**')
        
        if not all_data:
            st.error("❌ No se pudo extraer información de ninguna URL.")
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
