\documentclass[12pt,a4paper]{article}

% --------------------------------------------------
% Packages
% --------------------------------------------------
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[portuguese]{babel}
\usepackage{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{hyperref}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{caption}

% --------------------------------------------------
% Page setup
% --------------------------------------------------
\geometry{margin=2.5cm}
\setlength{\parskip}{0.8em}
\setlength{\parindent}{0pt}

% --------------------------------------------------
% Code style
% --------------------------------------------------
\lstset{
    language=Python,
    basicstyle=\ttfamily\small,
    keywordstyle=\color{blue},
    commentstyle=\color{gray},
    stringstyle=\color{teal},
    breaklines=true,
    frame=single
}

% --------------------------------------------------
% Document
% --------------------------------------------------
\begin{document}

\begin{titlepage}
    \centering
    \vspace*{3cm}
    
    {\Huge \textbf{Relatório de Projeto}}\\[0.5cm]
    {\Large \textbf{Inteligência Artificial}}\\[0.8cm]
    {\Large Previsão de Custos Médicos Anuais}\\[2cm]
    
    \textbf{Disciplina:} Inteligência Artificial\\
    \textbf{Semestre:} 3º Semestre\\
    \textbf{Data:} Janeiro de 2026\\
    \textbf{Autor:} Tino
    
    \vfill
\end{titlepage}

\tableofcontents
\newpage

% --------------------------------------------------
\section{Introdução}

\subsection{Contexto}

No âmbito da disciplina de Inteligência Artificial do 3º semestre, foi disponibilizado um dataset denominado \textbf{ECF\_2.xlsx}, contendo informações sobre pacientes de uma seguradora de saúde. O objetivo proposto foi explorar, analisar e aplicar técnicas de \textit{Machine Learning} para extrair valor preditivo dos dados.

\subsection{Objetivo do Projeto}

O objetivo principal do projeto foi:

\begin{quote}
\textbf{Desenvolver um modelo de Machine Learning capaz de prever o custo médico anual (\texttt{annual\_medical\_cost}) de um paciente, utilizando apenas informações disponíveis no momento da subscrição de um seguro de saúde.}
\end{quote}

Este objetivo simula um cenário real de negócio onde uma seguradora necessita estimar o custo esperado de um novo cliente para definir o prémio do seguro.

\subsection{Desafio Principal}

O principal desafio identificado foi a construção de um modelo \textbf{realista}, evitando o uso de variáveis que só existem após a utilização do seguro, fenómeno conhecido como \textbf{Data Leakage}.

% --------------------------------------------------
\section{Análise Exploratória do Dataset}

\subsection{Visão Geral}

O dataset \texttt{ECF\_2.xlsx} contém \textbf{9.950 registos} e \textbf{54 colunas}.

\textbf{TODO: Inserir imagem do df.info()}

\subsection{Categorias de Variáveis}

\subsubsection{Informações Demográficas}

\begin{longtable}{lll}
\toprule
Coluna & Descrição & Tipo \\
\midrule
person\_id & Identificador do paciente & ID \\
age & Idade & Numérico \\
gender & Género & Categórico \\
region & Região & Categórico \\
income & Rendimento anual & Numérico \\
education\_level & Nível de educação & Categórico \\
employment\_status & Situação profissional & Categórico \\
marital\_status & Estado civil & Categórico \\
household\_size & Agregado familiar & Numérico \\
dependents & Dependentes & Numérico \\
\bottomrule
\end{longtable}

\subsubsection{Informações Clínicas}

\begin{longtable}{lll}
\toprule
Coluna & Descrição & Tipo \\
\midrule
bmi & Índice de Massa Corporal & Numérico \\
smoker & Estado de fumador & Categórico \\
chronic\_count & Doenças crónicas & Numérico \\
medication\_count & Medicamentos & Numérico \\
cardiovascular\_disease & Doença cardiovascular & Binário \\
diabetes & Diabetes & Binário \\
cancer\_history & Histórico de cancro & Binário \\
kidney\_disease & Doença renal & Binário \\
copd & DPOC & Binário \\
\bottomrule
\end{longtable}

\subsubsection{Target}

\begin{center}
\textbf{annual\_medical\_cost — Custo médico anual do paciente}
\end{center}

\subsection{Análise de Correlações}

\textbf{TODO: Inserir heatmap de correlações}

As maiores correlações observadas foram:

\begin{verbatim}
total_claims_paid   0.97  (Data Leakage)
annual_premium      0.76  (Data Leakage)
risk_score          0.74  (Data Leakage)
chronic_count       0.23
age                 0.15
\end{verbatim}

As variáveis mais correlacionadas correspondem a \textbf{Data Leakage}, invalidando modelos que as utilizem.

% --------------------------------------------------
\section{Problema de Data Leakage}

\subsection{Definição}

\textbf{Data Leakage} ocorre quando o modelo utiliza informação que não estaria disponível no momento real da previsão.

\subsection{Variáveis Removidas}

\begin{longtable}{ll}
\toprule
Variável & Justificação \\
\midrule
claims\_count & Só existe após utilização \\
total\_claims\_paid & Correlação direta com target \\
risk\_score & Calculado com base no custo \\
annual\_premium & Output do modelo real \\
monthly\_premium & Derivado do prémio \\
\bottomrule
\end{longtable}

\subsection{Impacto nas Métricas}

\begin{verbatim}
Com Data Leakage:
R² ≈ 0.95  (modelo inútil)

Sem Data Leakage:
R² ≈ 0.14  (modelo realista)
\end{verbatim}

% --------------------------------------------------
\section{Feature Engineering}

\subsection{Risk Proxy}

\begin{lstlisting}
X['risk_proxy'] = (
    0.30 * (chronic_count / max_chronic) +
    0.25 * (age / 100) +
    0.20 * (bmi / 50).clip(0, 1) +
    0.15 * smoker_num +
    0.10 * (medication_count / max_med)
).clip(0, 1)
\end{lstlisting}

\subsection{Utilization Score}

\begin{lstlisting}
X['utilization_score'] = (
    0.6 * np.log1p(hospitalizations_last_3yrs) +
    0.4 * np.log1p(visits_last_year)
)
\end{lstlisting}

\subsection{Resumo das Features Criadas}

\begin{longtable}{lll}
\toprule
Feature & Descrição & Importância \\
\midrule
risk\_proxy & Proxy de risco clínico & Alta \\
utilization\_score & Utilização histórica & Média \\
procedure\_intensity & Intensidade de procedimentos & Média \\
hospital\_severity & Gravidade hospitalar & Média \\
\bottomrule
\end{longtable}

% --------------------------------------------------
\section{Modelos Testados}

Foram testados os seguintes modelos:
\begin{itemize}
\item Random Forest
\item Gradient Boosting
\item Deep Learning
\item Random Forest com deductible
\end{itemize}

\subsection{Random Forest}

\begin{lstlisting}
RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4,
    n_jobs=-1,
    random_state=42
)
\end{lstlisting}

% --------------------------------------------------
\section{Resultados e Análise}

\begin{longtable}{lcccc}
\toprule
Modelo & R² & MAE & WMAPE & Tempo (s) \\
\midrule
Random Forest & 0.1420 & 1897 & 62.4 & 1.3 \\
RF + Deductible & 0.1333 & 1907 & 62.8 & 1.5 \\
Deep Learning & 0.0992 & 1766 & 58.1 & 10.6 \\
Gradient Boosting & 0.0957 & 1921 & 63.2 & 6.5 \\
\bottomrule
\end{longtable}

% --------------------------------------------------
\section{Conclusões}

O objetivo do projeto foi alcançado com um modelo de Random Forest que explica \textbf{14.2\% da variância}. Apesar do valor de R² relativamente baixo, trata-se de um resultado \textbf{honesto e realista} para um cenário de produção sem Data Leakage.

Este trabalho demonstra que, em Machine Learning, \textbf{a integridade dos dados é mais importante do que métricas artificialmente elevadas}.

% --------------------------------------------------
\section{Anexos}

\subsection{Estrutura do Projeto}

\begin{verbatim}
projeto_IA/
├── ECF_2.xlsx
├── main_cost_prediction.py
├── RELATORIO.md
├── model_analysis.png
└── experiments/
\end{verbatim}

\subsection{Referências}

\begin{itemize}
\item Kaggle Medical Insurance Cost Prediction
\item Scikit-learn Documentation
\item TensorFlow/Keras Documentation
\item Towards Data Science – Data Leakage
\end{itemize}

\vfill
\begin{center}
\textit{Documento gerado em Janeiro de 2026}
\end{center}

\end{document}
