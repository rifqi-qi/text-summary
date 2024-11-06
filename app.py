import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import combinations

# Fungsi untuk membangun graf berdasarkan kata
def build_word_graph(text, top_n=5):
    # Pisahkan teks menjadi kalimat
    sentences = text.split('. ')
    
    # Hitung TF-IDF pada kata-kata
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # Inisialisasi graf
    G = nx.Graph()
    
    # Buat graf berdasarkan co-occurrence kata dalam setiap kalimat
    for sentence in sentences:
        words = [word for word in sentence.split() if word in feature_names]
        for word1, word2 in combinations(words, 2):
            if G.has_edge(word1, word2):
                G[word1][word2]['weight'] += 1
            else:
                G.add_edge(word1, word2, weight=1)
    
    # Hitung degree centrality untuk setiap kata
    degree_centrality = nx.degree_centrality(G)
    
    # Plot graf menggunakan degree centrality sebagai ukuran node
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=[v * 3000 for v in degree_centrality.values()], node_color='lightblue')
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='darkblue')
    plt.title("Graf Degree Centrality antar Kata")

    # Menampilkan graf di Streamlit
    st.pyplot(plt)

    # Menampilkan kata-kata penting berdasarkan degree centrality
    important_words = sorted(degree_centrality, key=degree_centrality.get, reverse=True)[:top_n]
    return important_words

# Streamlit App
st.title("Text Summarization dengan Degree Centrality Graf")

# Kolom input teks
input_text = st.text_area("Masukkan teks yang ingin diringkas:", height=200)

# Slider untuk menentukan jumlah kata penting
top_n = st.slider("Jumlah kata penting yang ditampilkan:", min_value=1, max_value=10, value=5)

# Tombol untuk memulai proses
if st.button("Proses Teks"):
    if input_text:
        # Menampilkan graf dan kata-kata penting
        important_words = build_word_graph(input_text, top_n=top_n)
        
        st.write("Kata-kata Penting Berdasarkan Degree Centrality:")
        st.write(important_words)
    else:
        st.write("Silakan masukkan teks terlebih dahulu.")
