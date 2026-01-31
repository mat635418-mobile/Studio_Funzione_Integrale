import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.integrate import quad
import warnings

# Configurazione pagina Streamlit
st.set_page_config(page_title="Studio Funzione Integrale v0.5", layout="wide")

# --- CLASSE DI ANALISI (Backend) ---
class IntegralAnalyzer:
    def __init__(self, func_str, variable='t'):
        self.t = sp.symbols(variable)
        self.func_str = func_str
        self.error = None
        
        try:
            # Parsing simbolico
            self.f_expr = sp.sympify(func_str)
            # Derivata prima di f (che è F'')
            self.f_prime_expr = sp.diff(self.f_expr, self.t)
            # Funzione numerica per integrazione (gestisce numpy array)
            self.f_numeric = sp.lambdify(self.t, self.f_expr, modules=['numpy'])
        except Exception as e:
            self.error = str(e)

    def check_limit(self, point, direction='+'):
        """Calcola limiti simbolici per asintoti."""
        try:
            dir_sym = '+' if direction == '+' else '-'
            lim = sp.limit(self.f_expr, self.t, point, dir=dir_sym)
            return lim
        except:
            return "N/A"

    def get_singularities(self):
        """Trova singolarità (punti dove f non è definita)."""
        try:
            # Cerca dove il denominatore è 0 (semplificazione euristica)
            denom = sp.fraction(self.f_expr)[1]
            if denom != 1:
                return sp.solve(denom, self.t)
            return []
        except:
            return []

    def compute_data(self, x0, x_range, num_points=400):
        """Calcola i dati numerici per il grafico."""
        xs = np.linspace(x_range[0], x_range[1], num_points)
        ys = []
        valid_xs = []
        
        # Integrazione punto per punto
        for x in xs:
            try:
                # Quad restituisce (valore, errore_stimato)
                # Limitiamo i cicli per evitare freeze su asintoti
                val, _ = quad(self.f_numeric, x0, x, limit=50)
                ys.append(val)
                valid_xs.append(x)
            except Exception:
                # Se l'integrale fallisce (es. singolarità non integrabile), interrompiamo o mettiamo NaN
                ys.append(np.nan)
                valid_xs.append(x)
                
        return np.array(valid_xs), np.array(ys)

# --- INTERFACCIA UTENTE (Frontend) ---

st.title("📈 Studio di Funzioni Integrali - v0.5")
st.markdown("Release Feb 2026 | *Analisi automatica con Python, SymPy e SciPy*")

# Sidebar per Input
with st.sidebar:
    st.header("Parametri")
    func_input = st.text_input("Funzione integranda f(t):", value="exp(t)/t")
    st.caption("Usa sintassi Python: `exp(t)`, `log(t)`, `sin(t)`, `t**2`")
    
    x0 = st.number_input("Punto iniziale (x0):", value=-1.0, step=0.5)
    
    st.subheader("Intervallo Grafico")
    col1, col2 = st.columns(2)
    x_min = col1.number_input("Min X", value=-4.0)
    x_max = col2.number_input("Max X", value=-0.1)
    
    st.info("💡 Suggerimento: Evita di includere singolarità non integrabili (come t=0 per e^t/t) all'interno dell'intervallo di integrazione se non sono gestite.")

# --- LOGICA PRINCIPALE ---

if func_input:
    analyzer = IntegralAnalyzer(func_input)
    
    if analyzer.error:
        st.error(f"Errore nella funzione: {analyzer.error}")
    else:
        # Visualizzazione LaTeX
        st.latex(r"F(x) = \int_{" + str(x0) + r"}^{x} " + sp.latex(analyzer.f_expr) + r"\, dt")
        
        # Calcolo Dati
        with st.spinner('Calcolo integrale e studio asintotico...'):
            xs, ys = analyzer.compute_data(x0, (x_min, x_max))
            
            # Calcolo Derivate simboliche per visualizzazione
            f_prime = analyzer.f_prime_expr
        
        # --- COLONNE ANALISI ---
        col_graph, col_info = st.columns([2, 1])
        
        with col_info:
            st.subheader("🔍 Analisi Analitica")
            
            # 1. Studio Derivate
            st.markdown("**Derivata Prima $F'(x) = f(x)$:**")
            st.latex(sp.latex(analyzer.f_expr))
            st.markdown("*Gli zeri di $f(x)$ sono i punti stazionari di $F(x)$.*")
            
            st.markdown("**Derivata Seconda $F''(x) = f'(x)$:**")
            st.latex(sp.latex(f_prime))
            st.markdown("*Gli zeri di $f'(x)$ sono i flessi di $F(x)$.*")
            
            # 2. Singolarità e Asintoti (Feature richiesta)
            st.markdown("---")
            st.markdown("**Analisi Singolarità:**")
            singularities = analyzer.get_singularities()
            if singularities:
                st.write(f"Punti problematici trovati: {singularities}")
                for s in singularities:
                    # Se la singolarità è reale
                    if s.is_real:
                        lim = analyzer.check_limit(s)
                        st.latex(r"\lim_{t \to " + f"{float(s):.2f}" + r"} f(t) = " + sp.latex(lim))
            else:
                st.write("Nessuna singolarità evidente trovata simbolicamente.")

        with col_graph:
            # --- PLOTTING AVANZATO ---
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Traccia principale
            # Filtriamo i NaN per evitare linee spezzate brutte
            mask = ~np.isnan(ys)
            ax.plot(xs[mask], ys[mask], label='F(x)', color='#1f77b4', linewidth=2.5)
            
            # 3. Studio del Segno (Feature richiesta: Colorazione)
            # Verde se F(x) > 0, Rosso se F(x) < 0
            ax.fill_between(xs[mask], ys[mask], 0, where=(ys[mask] > 0), 
                            color='green', alpha=0.1, label='F(x) > 0')
            ax.fill_between(xs[mask], ys[mask], 0, where=(ys[mask] < 0), 
                            color='red', alpha=0.1, label='F(x) < 0')
            
            # Elementi grafici
            ax.axhline(0, color='black', linewidth=1)
            ax.axvline(0, color='black', linewidth=1)
            ax.axvline(x0, color='orange', linestyle='--', linewidth=1.5, label=f'Start $x_0={x0}$')
            
            # Evidenziare asintoti verticali rilevati numericamente (valori enormi)
            # Se la derivata (f(x)) tende a infinito e l'integrale cresce rapido
            
            ax.set_title("Grafico della Funzione Integrale $F(x)$", fontsize=14)
            ax.set_xlabel("x")
            ax.set_ylabel("F(x)")
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend()
            
            st.pyplot(fig)
            
        # --- COMMENTO AUTOMATICO SUL GRAFICO ---
        st.subheader("📝 Note sull'andamento")
        if len(ys) > 0:
            last_y = ys[mask][-1]
            last_x = xs[mask][-1]
            
            trend = "crescente" if analyzer.f_numeric(last_x) > 0 else "decrescente"
            sign_note = "positivo" if last_y > 0 else "negativo"
            
            st.write(f"Nell'estremo destro dell'intervallo ($x \\approx {last_x:.1f}$), la funzione integrale è **{sign_note}** e ha un andamento locale **{trend}** (dato dal segno di $f(t)$).")
            
            # Controllo "Asintoto Verticale" euristico basato sul valore
            if abs(last_y) > 20: # Soglia arbitraria per demo
                st.warning("⚠️ Il valore dell'integrale è molto alto. Potrebbe esserci un asintoto verticale o una crescita verso infinito.")
