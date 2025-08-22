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
from datetime import datetime

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
    page_title="Análisis Competitivo de Productos",
    page_icon="🎯",
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
        except Exception:
            pass

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        try:
            nltk.download('stopwords', quiet=True)
        except Exception:
            pass

download_nltk_data()

def main():
    # CSS personalizado mejorado
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
        color: #155724;
    }
    .warning-message {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 0.5rem;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
        color: #856404;
    }
    .gap-card {
        background: #f8f9fa;
        border-left: 4px solid #FF6B6B;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header principal
    st.markdown('<h1 class="main-header">🎯 Análisis Competitivo de Productos</h1>', unsafe_allow_html=True)
    st.markdown("### Analiza productos de la competencia y encuentra oportunidades de mejora")
    
    # Mensaje de estado de librerías
    if not WORDCLOUD_AVAILABLE:
        st.info("ℹ️ WordCloud no está disponible. Las nubes de palabras se mostrarán como gráficos de barras.")
    
    # Información de ayuda mejorada
    with st.expander("📚 ¿Cómo usar esta herramienta?"):
        st.markdown("""
        ### 🎯 Flujo de Trabajo Recomendado
        
        1. **🔗 URL de Referencia**: Ingresa la URL de TU producto
        2. **🔍 URLs de Competencia**: Añade URLs de productos competidores
        3. **📊 Análisis Automático**: La herramienta extraerá y comparará:
           - Títulos y descripciones
           - Características y especificaciones
           - Filtros y categorías
           - Precios y posicionamiento
        4. **🎯 Análisis de GAPS**: Identifica qué le falta a tu producto
        5. **💡 Insights**: Obtén recomendaciones basadas en datos
        
        ### ✅ Sitios Compatibles
        - **Amazon** ⭐ Mejor compatibilidad
        - **eBay** ⭐ Muy buena compatibilidad
        - **AliExpress** ✅ Generalmente funciona
        - **Tiendas pequeñas** ✅ Menos restricciones
        
        ### 🚫 Sitios con Restricciones
        - MediaMarkt, PCComponentes, El Corte Inglés
        - Grandes retailers con protección anti-bot
        
        ### 💡 Tips Pro
        - Usa el **modo agresivo** para sitios difíciles
        - Aumenta el **delay** entre requests para evitar bloqueos
        - Analiza productos similares, no idénticos
        - Combina con Google Shopping para vista de mercado
        """)
    
    # Sidebar mejorado
    st.sidebar.header("⚙️ Configuración")
    
    # Opciones de análisis
    st.sidebar.subheader("📋 Tipos de Análisis")
    analyze_terms = st.sidebar.checkbox("🔤 Términos clave", value=True)
    analyze_filters = st.sidebar.checkbox("🎛️ Filtros y categorías", value=True)
    analyze_features = st.sidebar.checkbox("⭐ Características", value=True)
    analyze_gaps = st.sidebar.checkbox("🎯 Análisis de GAPS", value=True)
    analyze_pricing = st.sidebar.checkbox("💰 Análisis de precios", value=True)
    
    if WORDCLOUD_AVAILABLE:
        show_wordcloud = st.sidebar.checkbox("☁️ Nube de palabras", value=True)
    else:
        show_wordcloud = False
    
    st.sidebar.markdown("---")
    
    # Configuración avanzada
    st.sidebar.subheader("🔧 Configuración Avanzada")
    
    top_n = st.sidebar.slider("📊 Top N resultados", 5, 50, 20)
    delay = st.sidebar.slider("⏱️ Delay entre requests (seg)", 0.5, 5.0, 2.0, 0.5)
    
    st.sidebar.markdown("**🛡️ Anti-detección:**")
    retry_403 = st.sidebar.checkbox("🔄 Reintentar bloqueados", value=True)
    aggressive_mode = st.sidebar.checkbox("🚀 Modo agresivo", value=False)
    rotate_headers = st.sidebar.checkbox("🔄 Rotar User-Agents", value=False)
    
    if aggressive_mode:
        delay = max(delay, 3.0)
    
    # Pestañas principales
    tab1, tab2, tab3 = st.tabs(["📊 Análisis de URLs", "🛒 Google Shopping", "📈 Comparación"])
    
    with tab1:
        st.header("🔗 Análisis de Productos por URL")
        
        # URL de referencia (nuevo)
        st.subheader("🎯 Producto de Referencia (Tu Producto)")
        reference_url = st.text_input(
            "URL de tu producto (opcional - para análisis de gaps):",
            placeholder="https://tu-tienda.com/tu-producto",
            help="Esta será la referencia para comparar con la competencia"
        )
        
        # URLs de competencia
        st.subheader("🔍 Productos de la Competencia")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            urls_input = st.text_area(
                "URLs de productos competidores (una por línea):",
                height=200,
                placeholder="""https://www.amazon.es/dp/B08N5WRWNW
https://www.ebay.es/itm/123456789
https://www.aliexpress.com/item/1005001234567890.html""",
                help="Ingresa las URLs de productos que quieres analizar"
            )
        
        with col2:
            st.markdown("**✅ Compatibles:**")
            st.success("amazon.es/com")
            st.success("ebay.es/com")
            st.success("aliexpress.com")
            st.markdown("**⚠️ Difíciles:**")
            st.warning("mediamarkt.es")
            st.warning("pccomponentes.com")
        
        # Validación de URLs
        if urls_input.strip() or reference_url.strip():
            all_urls = []
            
            if reference_url.strip() and reference_url.startswith(('http://', 'https://')):
                all_urls.append(('reference', reference_url.strip()))
            
            if urls_input.strip():
                comp_urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
                for url in comp_urls:
                    if url.startswith(('http://', 'https://')):
                        all_urls.append(('competitor', url))
            
            if all_urls:
                st.success(f"✅ {len(all_urls)} URLs válidas detectadas")
        
        # Botón de análisis
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button(
                "🚀 INICIAR ANÁLISIS", 
                type="primary", 
                use_container_width=True,
                disabled=not (urls_input.strip() or reference_url.strip())
            )
        
        if analyze_button:
            analyzer = ProductBenchmarkAnalyzer()
            
            # Progreso
            st.markdown("### 🔄 Procesando URLs...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Métricas
            col1, col2, col3 = st.columns(3)
            with col1:
                success_metric = st.metric("✅ Exitosos", 0)
            with col2:
                failed_metric = st.metric("❌ Fallidos", 0)
            with col3:
                total_metric = st.metric("📊 Total", len(all_urls))
            
            reference_data = None
            competitor_data = []
            failed_count = 0
            success_count = 0
            
            # Procesar cada URL
            for i, (url_type, url) in enumerate(all_urls):
                status_text.markdown(f'🔍 **Procesando {url_type} {i+1}/{len(all_urls)}**  \n`{url[:70]}...`')
                
                if i > 0:
                    time.sleep(delay * 1.5 if aggressive_mode else delay)
                
                data = analyzer.extract_content_from_url(url, rotate_headers)
                
                if data:
                    if url_type == 'reference':
                        reference_data = data
                    else:
                        competitor_data.append(data)
                    success_count += 1
                    success_metric.metric("✅ Exitosos", success_count)
                else:
                    failed_count += 1
                    failed_metric.metric("❌ Fallidos", failed_count)
                    
                    # Retry si está habilitado
                    if retry_403:
                        status_text.markdown(f'🔄 **Reintentando...**')
                        time.sleep(5)
                        retry_data = analyzer.extract_content_from_url(url, True)
                        if retry_data:
                            if url_type == 'reference':
                                reference_data = retry_data
                            else:
                                competitor_data.append(retry_data)
                            success_count += 1
                            failed_count -= 1
                            success_metric.metric("✅ Exitosos", success_count)
                            failed_metric.metric("❌ Fallidos", failed_count)
                
                progress_bar.progress((i + 1) / len(all_urls))
            
            status_text.markdown('✅ **Análisis completado**')
            
            # Guardar datos en session state
            st.session_state['reference_data'] = reference_data
            st.session_state['competitor_data'] = competitor_data
            st.session_state['all_data'] = [reference_data] if reference_data else [] + competitor_data
            
            if not st.session_state['all_data']:
                st.error("❌ No se pudo extraer información de ninguna URL.")
                return
            
            # Mensaje de éxito
            st.markdown(f"""
            <div class="success-message">
                <strong>🎉 ¡Análisis completado!</strong><br>
                Se procesaron <strong>{success_count}</strong> de <strong>{len(all_urls)}</strong> productos
            </div>
            """, unsafe_allow_html=True)
            
            # Pestañas de resultados
            result_tabs = st.tabs([
                "📊 Resumen", 
                "🎯 Análisis de GAPS",
                "🔤 Términos", 
                "🎛️ Filtros", 
                "⭐ Características",
                "💰 Precios",
                "📈 Visualizaciones",
                "💾 Exportar"
            ])
            
            with result_tabs[0]:  # Resumen
                st.header("📊 Resumen del Análisis")
                
                # Métricas principales
                col1, col2, col3, col4 = st.columns(4)
                
                all_data = st.session_state['all_data']
                
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
                
                # Tabla resumen
                summary_data = []
                for i, data in enumerate(all_data):
                    is_reference = (i == 0 and reference_data)
                    summary_data.append({
                        'Tipo': '🎯 Referencia' if is_reference else f'🔍 Competidor {i}',
                        'Dominio': data.get('domain', 'N/A'),
                        'Título': data.get('title', 'Sin título')[:60] + '...',
                        'Precio': data.get('price', 'N/A'),
                        'Características': len(data.get('features', [])),
                        'Especificaciones': len(data.get('specifications', {})),
                        'Filtros': len(data.get('filters', []))
                    })
                
                df_summary = pd.DataFrame(summary_data)
                st.dataframe(df_summary, use_container_width=True, hide_index=True)
            
            with result_tabs[1]:  # Análisis de GAPS
                st.header("🎯 Análisis de GAPS")
                
                if reference_data and competitor_data:
                    gaps = analyzer.analyze_gaps(reference_data, competitor_data)
                    
                    # Características únicas de la competencia
                    if gaps['unique_competitor_features']:
                        st.subheader("⚡ Características que tiene la competencia")
                        st.markdown('<div class="warning-message">', unsafe_allow_html=True)
                        st.markdown("**Oportunidades de mejora detectadas:**")
                        for feature in gaps['unique_competitor_features'][:10]:
                            st.markdown(f"• {feature}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Especificaciones faltantes
                    if gaps['missing_specs']:
                        st.subheader("📋 Especificaciones faltantes")
                        cols = st.columns(3)
                        for i, spec in enumerate(gaps['missing_specs'][:12]):
                            cols[i % 3].warning(f"📌 {spec}")
                    
                    # Filtros que usa la competencia
                    if gaps['missing_filters']:
                        st.subheader("🎛️ Filtros adicionales en competencia")
                        st.info("Considera añadir estos filtros a tu tienda:")
                        filter_df = pd.DataFrame(
                            {'Filtro': gaps['missing_filters'][:20]},
                            index=range(1, min(21, len(gaps['missing_filters'])+1))
                        )
                        st.dataframe(filter_df, use_container_width=True)
                    
                    # Análisis de precio
                    if gaps['price_difference']:
                        st.subheader("💰 Análisis de Precio")
                        price_data = gaps['price_difference']
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Tu Precio", f"{price_data['reference']:.2f}€")
                        with col2:
                            st.metric("Promedio Competencia", f"{price_data['competitors_avg']:.2f}€")
                        with col3:
                            diff_color = "🟢" if price_data['difference'] < 0 else "🔴"
                            st.metric(
                                "Diferencia",
                                f"{abs(price_data['difference']):.2f}€",
                                f"{diff_color} {abs(price_data['percentage']):.1f}%"
                            )
                        
                        if price_data['percentage'] > 20:
                            st.warning("⚠️ Tu precio es significativamente mayor que la competencia")
                        elif price_data['percentage'] < -20:
                            st.info("💡 Tu precio es muy competitivo, podrías considerar ajustarlo")
                
                else:
                    st.info("💡 Añade una URL de referencia y URLs de competencia para ver el análisis de gaps")
            
            # Resto de pestañas con análisis tradicionales...
            with result_tabs[2]:  # Términos
                if analyze_terms:
                    st.header("🔤 Términos Más Relevantes")
                    terms = analyzer.analyze_terms(all_data)
                    top_terms = terms.most_common(top_n)
                    
                    if top_terms:
                        df_terms = pd.DataFrame(top_terms, columns=['Término', 'Frecuencia'])
                        
                        fig = px.bar(
                            df_terms, 
                            x='Frecuencia', 
                            y='Término',
                            orientation='h',
                            color='Frecuencia',
                            color_continuous_scale='viridis',
                            title="Términos clave más frecuentes"
                        )
                        fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
            
            with result_tabs[7]:  # Exportar
                st.header("💾 Exportar Resultados")
                
                # Preparar datos para exportación
                export_data = []
                
                for i, data in enumerate(all_data):
                    is_reference = (i == 0 and reference_data)
                    export_data.append({
                        'Tipo': 'Referencia' if is_reference else 'Competidor',
                        'URL': data.get('url', ''),
                        'Dominio': data.get('domain', ''),
                        'Título': data.get('title', ''),
                        'Descripción': data.get('description', '')[:500],
                        'Precio': data.get('price', ''),
                        'Características': ' | '.join(data.get('features', [])),
                        'Especificaciones': json.dumps(data.get('specifications', {}), ensure_ascii=False),
                        'Filtros': ' | '.join(data.get('filters', [])),
                        'Categorías': ' | '.join(data.get('categories', [])),
                        'Fecha_Extracción': data.get('extracted_at', '')
                    })
                
                df_export = pd.DataFrame(export_data)
                
                # Botón de descarga
                csv = df_export.to_csv(index=False, encoding='utf-8')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="📥 Descargar Análisis Completo (CSV)",
                        data=csv,
                        file_name=f"analisis_competitivo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    # Exportar solo gaps si existe
                    if reference_data and competitor_data:
                        gaps = analyzer.analyze_gaps(reference_data, competitor_data)
                        gaps_text = f"""ANÁLISIS DE GAPS - {datetime.now().strftime('%Y-%m-%d %H:%M')}
                        
CARACTERÍSTICAS ÚNICAS DE COMPETENCIA:
{chr(10).join('• ' + f for f in gaps['unique_competitor_features'][:20])}

ESPECIFICACIONES FALTANTES:
{chr(10).join('• ' + s for s in gaps['missing_specs'][:20])}

FILTROS ADICIONALES EN COMPETENCIA:
{chr(10).join('• ' + f for f in gaps['missing_filters'][:20])}

ANÁLISIS DE PRECIO:
{json.dumps(gaps['price_difference'], indent=2) if gaps['price_difference'] else 'No disponible'}
"""
                        st.download_button(
                            label="📥 Descargar Análisis de GAPS (TXT)",
                            data=gaps_text,
                            file_name=f"gaps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
    
    with tab2:  # Google Shopping
        st.header("🛒 Análisis con Google Shopping")
        st.info("💡 Analiza el mercado completo sin restricciones de sitios web")
        
        search_query = st.text_input(
            "¿Qué producto quieres analizar?",
            placeholder="Ejemplo: auriculares bluetooth deportivos",
            help="Describe el producto para buscar en el mercado"
        )
        
        num_results = st.slider("Número de resultados", 5, 30, 15)
        
        if st.button("🔍 Buscar en Google Shopping", type="primary", disabled=not search_query):
            shopping_analyzer = GoogleShoppingAnalyzer()
            
            with st.spinner("Buscando productos..."):
                products = shopping_analyzer.search_products_free(search_query, num_results)
            
            if products:
                st.success(f"✅ Se encontraron {len(products)} productos")
                
                analysis = shopping_analyzer.analyze_shopping_data(products)
                
                # Sub-pestañas para resultados
                shop_tabs = st.tabs(["📋 Productos", "📊 Tiendas", "💰 Precios", "🔤 Términos"])
                
                with shop_tabs[0]:
                    st.subheader("Productos Encontrados")
                    for i, product in enumerate(products[:10], 1):
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.markdown(f"**{i}. {product.get('title', 'Sin título')[:80]}**")
                        with col2:
                            st.markdown(f"💰 **{product.get('price', 'N/A')}**")
                        with col3:
                            st.markdown(f"🏪 {product.get('source', 'N/A')}")
                        st.divider()
                
                with shop_tabs[1]:
                    if analysis['sources']:
                        sources_df = pd.DataFrame(
                            list(analysis['sources'].items()), 
                            columns=['Tienda', 'Productos']
                        ).sort_values('Productos', ascending=False)
                        
                        fig = px.bar(
                            sources_df, 
                            x='Productos', 
                            y='Tienda',
                            orientation='h',
                            title="Distribución por Tienda"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with shop_tabs[2]:
                    if analysis.get('price_ranges'):
                        price_info = analysis['price_ranges']
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("💵 Mínimo", f"{price_info['min']:.2f}€")
                        with col2:
                            st.metric("💸 Máximo", f"{price_info['max']:.2f}€")
                        with col3:
                            st.metric("📊 Promedio", f"{price_info['avg']:.2f}€")
                        with col4:
                            st.metric("🔢 Con Precio", price_info['count'])
                
                with shop_tabs[3]:
                    if analysis['common_terms']:
                        terms_data = analysis['common_terms'].most_common(25)
                        terms_df = pd.DataFrame(terms_data, columns=['Término', 'Frecuencia'])
                        
                        fig = px.bar(
                            terms_df.head(15),
                            x='Frecuencia',
                            y='Término',
                            orientation='h',
                            title="Términos más frecuentes",
                            color='Frecuencia',
                            color_continuous_scale='plasma'
                        )
                        fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("No se encontraron productos. Intenta con otro término.")
    
    with tab3:  # Comparación
        st.header("📈 Comparación Visual")
        
        if 'all_data' in st.session_state and st.session_state['all_data']:
            all_data = st.session_state['all_data']
            
            # Crear comparación visual
            st.subheader("🔍 Matriz de Comparación")
            
            comparison_matrix = []
            for data in all_data:
                comparison_matrix.append({
                    'Producto': data.get('title', 'Sin título')[:50],
                    'Precio': 1 if data.get('price') else 0,
                    'Descripción': len(data.get('description', '')) > 100,
                    'Características': len(data.get('features', [])),
                    'Especificaciones': len(data.get('specifications', {})),
                    'Imágenes': len(data.get('images', [])),
                    'Categorías': len(data.get('categories', []))
                })
            
            df_comparison = pd.DataFrame(comparison_matrix)
            
            # Heatmap de características
            fig = px.imshow(
                df_comparison.set_index('Producto').T,
                labels=dict(x="Productos", y="Atributos", color="Valor"),
                aspect="auto",
                color_continuous_scale="RdYlGn",
                title="Mapa de Calor - Completitud de Información"
            )
            # Ajustar el ángulo de las etiquetas del eje x
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Gráfico de radar para comparación
            if len(all_data) <= 5:  # Solo mostrar radar si hay 5 o menos productos
                st.subheader("🎯 Comparación Radar")
                
                categories = ['Características', 'Especificaciones', 'Imágenes', 'Categorías', 'Filtros']
                
                fig = go.Figure()
                
                for i, data in enumerate(all_data):
                    values = [
                        len(data.get('features', [])),
                        len(data.get('specifications', {})),
                        len(data.get('images', [])),
                        len(data.get('categories', [])),
                        len(data.get('filters', []))
                    ]
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name=f"Producto {i+1}"
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, max([
                                max(len(d.get('features', [])), 
                                    len(d.get('specifications', {})),
                                    len(d.get('images', [])),
                                    len(d.get('categories', [])),
                                    len(d.get('filters', [])))
                                for d in all_data
                            ])]
                        )
                    ),
                    showlegend=True,
                    title="Comparación de Completitud por Producto"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Insights automáticos
            st.subheader("💡 Insights Automáticos")
            
            insights = []
            
            # Análisis de completitud
            completeness_scores = []
            for data in all_data:
                score = 0
                score += 1 if data.get('title') else 0
                score += 1 if data.get('description') else 0
                score += 1 if data.get('price') else 0
                score += min(len(data.get('features', [])) / 5, 1)
                score += min(len(data.get('specifications', {})) / 5, 1)
                score += min(len(data.get('images', [])) / 3, 1)
                completeness_scores.append(score / 6 * 100)
            
            best_product_idx = completeness_scores.index(max(completeness_scores))
            worst_product_idx = completeness_scores.index(min(completeness_scores))
            
            insights.append(f"📊 **Producto más completo**: Producto {best_product_idx + 1} ({completeness_scores[best_product_idx]:.1f}% completitud)")
            insights.append(f"⚠️ **Producto menos completo**: Producto {worst_product_idx + 1} ({completeness_scores[worst_product_idx]:.1f}% completitud)")
            
            # Análisis de precios
            prices = []
            for data in all_data:
                if data.get('price'):
                    price_match = re.search(r'[\d,]+\.?\d*', data.get('price', '').replace(',', ''))
                    if price_match:
                        try:
                            prices.append(float(price_match.group()))
                        except:
                            pass
            
            if prices:
                avg_price = sum(prices) / len(prices)
                insights.append(f"💰 **Precio promedio**: {avg_price:.2f}€")
                insights.append(f"💵 **Rango de precios**: {min(prices):.2f}€ - {max(prices):.2f}€")
            
            # Mostrar insights
            for insight in insights:
                st.info(insight)
            
            # Recomendaciones
            st.subheader("🎯 Recomendaciones")
            
            recommendations = []
            
            if 'reference_data' in st.session_state and st.session_state['reference_data']:
                ref_data = st.session_state['reference_data']
                ref_score = completeness_scores[0]
                
                if ref_score < 70:
                    recommendations.append("🔴 **Urgente**: Tu producto tiene poca información. Añade más descripciones y características.")
                elif ref_score < 85:
                    recommendations.append("🟡 **Importante**: Tu producto está bien pero puede mejorar. Considera añadir más especificaciones técnicas.")
                else:
                    recommendations.append("🟢 **Excelente**: Tu producto tiene información muy completa.")
                
                if not ref_data.get('price'):
                    recommendations.append("💰 **Precio**: Considera mostrar el precio claramente en la página del producto.")
                
                if len(ref_data.get('images', [])) < 3:
                    recommendations.append("📸 **Imágenes**: Añade más imágenes del producto (mínimo 3-5).")
            
            if not recommendations:
                recommendations.append("💡 Añade una URL de referencia para obtener recomendaciones personalizadas.")
            
            for rec in recommendations:
                st.markdown(rec)
        
        else:
            st.info("👆 Primero realiza un análisis en la pestaña 'Análisis de URLs' para ver comparaciones.")
    
    # Footer con información adicional
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🎯 <strong>Análisis Competitivo de Productos</strong> v2.0</p>
        <p>Desarrollado para ayudarte a mejorar tu estrategia de producto</p>
        <p>💡 Tip: Analiza regularmente a tu competencia para mantenerte actualizado</p>
    </div>
    """, unsafe_allow_html=True)

class GoogleShoppingAnalyzer:
    """Obtiene resultados básicos de Google Shopping y realiza un análisis simple."""

    def __init__(self):
        # Cabeceras aleatorias para evitar bloqueos sencillos
        self.headers_options = [
            {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept-Language': 'es-ES,es;q=0.9,en;q=0.8'
            },
            {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept-Language': 'es,en-US;q=0.7,en;q=0.3'
            }
        ]

    def _get_headers(self):
        """Devuelve cabeceras aleatorias."""
        return random.choice(self.headers_options)

    def search_products_free(self, query, num_results=10):
        """Busca productos en Google Shopping sin necesidad de API.

        Parameters
        ----------
        query: str
            Término de búsqueda.
        num_results: int
            Número aproximado de resultados a recuperar.
        """
        url = f"https://www.google.com/search?tbm=shop&q={quote_plus(query)}&num={num_results}"
        products = []
        try:
            resp = requests.get(url, headers=self._get_headers(), timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")

            for item in soup.select("div.sh-dgr__content"):
                title_el = item.select_one("h3")
                price_el = item.select_one("span.a8Pemb")
                source_el = item.select_one("div.aULzUe span")

                products.append({
                    "title": title_el.get_text(strip=True) if title_el else "",
                    "price": price_el.get_text(strip=True) if price_el else "",
                    "source": source_el.get_text(strip=True) if source_el else ""
                })

                if len(products) >= num_results:
                    break
        except Exception:
            # En caso de fallo, devolver lista vacía para evitar errores en la app
            return []

        return products

    def analyze_shopping_data(self, products):
        """Realiza un análisis simple de los productos obtenidos."""
        if not products:
            return {}

        # Conteo de tiendas
        store_counts = Counter(p.get("source", "Desconocido") for p in products)

        # Conversión de precios a valores numéricos
        prices = []
        for p in products:
            price = p.get("price")
            if price:
                value = re.sub(r"[^0-9.,]", "", price).replace(",", ".")
                try:
                    prices.append(float(value))
                except Exception:
                    pass

        # Términos más frecuentes de los títulos
        terms = Counter()
        for p in products:
            terms.update(re.findall(r"\w+", p.get("title", "").lower()))

        return {
            "stores": dict(store_counts),
            "price_stats": {
                "min": min(prices) if prices else None,
                "max": max(prices) if prices else None,
                "avg": sum(prices) / len(prices) if prices else None,
            },
            "terms": dict(terms.most_common(20)),
        }

class ProductBenchmarkAnalyzer:
    def __init__(self):
        """Inicializa el analizador con stopwords mejoradas"""
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
        self.headers_options = [
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
        
    def extract_content_from_url(self, url, rotate_headers=False):
        """Extrae contenido relevante de una URL de producto"""
        try:
            # Seleccionar headers
            if rotate_headers:
                headers = random.choice(self.headers_options)
            else:
                headers = self.headers_options[0]
            
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
                'domain': urlparse(url).netloc,
                'title': self._extract_title(soup),
                'description': self._extract_description(soup),
                'features': self._extract_features(soup),
                'specifications': self._extract_specifications(soup),
                'price': self._extract_price(soup),
                'filters': self._extract_filters(soup),
                'categories': self._extract_categories(soup),
                'images': self._extract_images(soup),
                'extracted_at': datetime.now().isoformat()
            }
            
            return product_data
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                domain = urlparse(url).netloc
                st.warning(f"🚫 Acceso denegado a {domain}")
                self._suggest_alternatives(domain)
            else:
                st.warning(f"⚠️ Error HTTP {e.response.status_code} con {url[:50]}...")
            return None
        except requests.exceptions.RequestException as e:
            st.warning(f"⚠️ Error de conexión con {url[:50]}...: {str(e)}")
            return None
        except Exception as e:
            st.warning(f"⚠️ Error procesando {url[:50]}...: {str(e)}")
            return None
    
    def _suggest_alternatives(self, domain):
        """Sugiere alternativas para sitios bloqueados"""
        alternatives = {
            'mediamarkt': "💡 **Alternativa para MediaMarkt:** Busca el mismo producto en Amazon o eBay",
            'pccomponentes': "💡 **Alternativa para PCComponentes:** Prueba con Amazon o tiendas especializadas",
            'elcorteingles': "💡 **Alternativa para El Corte Inglés:** Busca en Amazon o tiendas del fabricante"
        }
        
        for site, message in alternatives.items():
            if site in domain.lower():
                st.info(message)
                break
    
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
        spec_selectors = [
            'table[class*="spec"]',
            'table[class*="detail"]',
            'table[class*="tech"]',
            'dl[class*="spec"]',
            'table'
        ]
        
        for selector in spec_selectors:
            elements = soup.select(selector)
            for element in elements:
                if element.name == 'table':
                    rows = element.find_all('tr')
                    for row in rows:
                        cells = row.find_all(['td', 'th'])
                        if len(cells) >= 2:
                            key = cells[0].get_text().strip()
                            value = cells[1].get_text().strip()
                            if key and value and len(key) < 100 and len(value) < 200:
                                specs[key] = value
                elif element.name == 'dl':
                    dts = element.find_all('dt')
                    dds = element.find_all('dd')
                    for dt, dd in zip(dts, dds):
                        key = dt.get_text().strip()
                        value = dd.get_text().strip()
                        if key and value:
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
            'span[itemprop="price"]',
            'meta[itemprop="price"]'
        ]
        
        for selector in price_selectors:
            elements = soup.select(selector)
            for element in elements:
                if element.name == 'meta':
                    price = element.get('content', '')
                    if price:
                        return price
                else:
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
            '[type="checkbox"] + label',
            '[class*="refinement"]'
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
            '.breadcrumb a',
            'nav[aria-label*="breadcrumb"] a'
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
    
    def _extract_images(self, soup):
        """Extrae URLs de imágenes del producto"""
        images = []
        
        image_selectors = [
            'img[class*="product"]',
            'img[data-testid*="product"]',
            '[class*="gallery"] img',
            '[class*="image"] img',
            'picture img'
        ]
        
        for selector in image_selectors:
            elements = soup.select(selector)
            for element in elements:
                src = element.get('src') or element.get('data-src')
                if src and not any(x in src.lower() for x in ['placeholder', 'loading', 'spinner']):
                    images.append(src)
        
        return list(set(images))[:10]
    
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
    
    def analyze_gaps(self, reference_data, comparison_data):
        """Analiza gaps entre producto de referencia y competencia"""
        gaps = {
            'missing_features': [],
            'missing_specs': [],
            'missing_filters': [],
            'unique_competitor_features': [],
            'price_difference': None,
            'category_differences': []
        }
        
        if not reference_data or not comparison_data:
            return gaps
        
        # Analizar características faltantes
        ref_features = set([f.lower() for f in reference_data.get('features', [])])
        
        for comp_data in comparison_data:
            comp_features = set([f.lower() for f in comp_data.get('features', [])])
            
            # Características que tiene la competencia pero no la referencia
            unique_comp = comp_features - ref_features
            gaps['unique_competitor_features'].extend(list(unique_comp))
            
            # Características que tiene la referencia pero no la competencia
            missing = ref_features - comp_features
            gaps['missing_features'].extend(list(missing))
        
        # Analizar especificaciones faltantes
        ref_specs = set(reference_data.get('specifications', {}).keys())
        
        for comp_data in comparison_data:
            comp_specs = set(comp_data.get('specifications', {}).keys())
            missing_specs = ref_specs - comp_specs
            gaps['missing_specs'].extend(list(missing_specs))
        
        # Analizar filtros
        ref_filters = set(reference_data.get('filters', []))
        all_comp_filters = set()
        
        for comp_data in comparison_data:
            all_comp_filters.update(comp_data.get('filters', []))
        
        gaps['missing_filters'] = list(all_comp_filters - ref_filters)
        
        # Analizar diferencias de precio
        ref_price = self._extract_price_value(reference_data.get('price', ''))
        if ref_price:
            comp_prices = []
            for comp_data in comparison_data:
                comp_price = self._extract_price_value(comp_data.get('price', ''))
                if comp_price:
                    comp_prices.append(comp_price)
            
            if comp_prices:
                avg_comp_price = sum(comp_prices) / len(comp_prices)
                gaps['price_difference'] = {
                    'reference': ref_price,
                    'competitors_avg': avg_comp_price,
                    'difference': ref_price - avg_comp_price,
                    'percentage': ((ref_price - avg_comp_price) / avg_comp_price) * 100
                }
        
        # Eliminar duplicados
        gaps['missing_features'] = list(set(gaps['missing_features']))
        gaps['missing_specs'] = list(set(gaps['missing_specs']))
        gaps['unique_competitor_features'] = list(set(gaps['unique_competitor_features']))
        
        return gaps
    
    def _extract_price_value(self, price_text):
        """Extrae el valor numérico del precio"""
        if not price_text:
            return None
        
        # Buscar números en el texto del precio
        price_match = re.search(r'[\d,]+\.?\d*', price_text.replace(',', ''))
        if price_match:
            try:
                return float(price_match.group())
            except:
                return None
        return None


if __name__ == "__main__":
    main()
