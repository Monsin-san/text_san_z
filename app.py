import plotly.express as px
import streamlit as st
import pandas as pd
from PIL import Image
from janome.tokenizer import Tokenizer
import base64
import re
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from janome.charfilter import *
from janome.tokenfilter import *
import os
from collections import Counter
import numpy as np
import io
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import tempfile
import chardet
import networkx as nx
from networkx.algorithms.community import louvain_communities

st.write("最終更新日：2024年＊月＊日")

def make_wordcloud(word_freq):
    # WordCloud オブジェクトの生成
    fpath = "MEIRYO.TTC"  # フォントパスの指定（システムによって異なる場合があります）
    wordcloud = WordCloud(font_path=fpath, width=800, height=400, background_color="white", collocations=False, max_words=100).generate_from_frequencies(word_freq)
    
    # ワードクラウドを表示するための図を作成
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

def load_file(file):
    if file is not None:
        file.seek(0)
        return pd.read_csv(file)

def create_user_dic(df):
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', encoding='utf-8') as f:
        for word in df['word']:
            f.write(f"{word},-1,-1,-100000,強制抽出語,一般,*,*,*,*,{word},{word},{word}\n")
        temp_path = f.name
    return temp_path

def create_synonym_map(df_synonyms):
    synonym_map = {}
    for index, row in df_synonyms.iterrows():
        key_word = row.iloc[0]
        synonyms = row.iloc[1:].dropna().tolist()
        for synonym in synonyms:
            synonym_map[synonym] = key_word
    return synonym_map

def preprocess_text(text):
    text = text.translate(str.maketrans({chr(0xFF01 + i): chr(0x21 + i) for i in range(94)})).lower()
    text = text.replace(" ", "").replace("　", "")
    return text

def analyze_text(text, user_dic_path=None, exclusion_words=[], synonym_map={}, pos_filters=None):
    text = preprocess_text(text)
    if user_dic_path:
        tokenizer = Tokenizer(udic=user_dic_path, udic_enc='utf8')
    else:
        tokenizer = Tokenizer()
    if not pos_filters or "すべて" in pos_filters:
        return [token for token in tokenizer.tokenize(text) if token.surface not in exclusion_words]
    else:
        return [token for token in tokenizer.tokenize(text) if token.part_of_speech.split(',')[0] in pos_filters and token.surface not in exclusion_words]

def count_words(tokens):
    word_count = {}
    for token in tokens:
        word = token.base_form if token.base_form != '*' else token.surface
        if word not in word_count:
            word_count[word] = 1
        else:
            word_count[word] += 1
    return word_count


def jaccard_similarity(set1, set2):
    return len(set1.intersection(set2)) / len(set1.union(set2))

def make_network_with_jaccard(raw_texts, jaccard_threshold, min_word_freq, min_cooccurrence, top_n_edges, selected_pos, user_dic_path=None, exclusion_words=[], synonym_map={}, max_node_size=1000):
    font = FontProperties(fname="MEIRYO.TTC")
    tokenizer = Tokenizer(udic=user_dic_path, udic_enc='utf8')
    
    all_sentences = []
    for text in raw_texts:
        sentences = re.split(r'。', text)
        all_sentences.extend(sentences)
    
    word_to_sentences = defaultdict(set)
    for sentence in all_sentences:
        for token in tokenizer.tokenize(sentence):
            pos = token.part_of_speech.split(',')[0]
            if pos not in selected_pos and 'すべて' not in selected_pos:
                continue  # 選択された品詞以外は無視
            word = synonym_map.get(token.surface, token.surface)
            if word in exclusion_words:
                continue  # 排除語は無視
            if len(word) == 1 or re.match(r'^\d+$', word):
                continue
            word_to_sentences[word].add(sentence)
                
    # Filter words with frequency less than min_word_freq
    word_to_sentences = {word: sent_set for word, sent_set in word_to_sentences.items() if len(sent_set) >= min_word_freq}

    G = nx.Graph()
    words = list(word_to_sentences.keys())
    edge_data = []

    for i in range(len(words)):
        for j in range(i+1, len(words)):
            word1 = words[i]
            word2 = words[j]
            jaccard_coeff = jaccard_similarity(word_to_sentences[word1], word_to_sentences[word2])
            if jaccard_coeff >= jaccard_threshold and len(word_to_sentences[word1].intersection(word_to_sentences[word2])) >= min_cooccurrence:
                edge_data.append((word1, word2, jaccard_coeff))
    
    edge_data.sort(key=lambda x: x[2], reverse=True)
    edge_data = edge_data[:top_n_edges]

    for word1, word2, weight in edge_data:
        G.add_edge(word1, word2, weight=weight)

    if not G.edges():
        return None

    # コミュニティ検出をLouvain法で実施
    communities = louvain_communities(G, seed=42)
    partition = {node: cid for cid, community in enumerate(communities) for node in community}
    color_map = [partition[node] for node in G.nodes()]
    
    pos = nx.spring_layout(G, k=0.8, iterations=100, scale=2.0, seed=30)
    edge_width = [G[u][v]['weight'] * 3 for u, v in G.edges()]
    
    max_node_size = 10000
    
    # Calculate node sizes
    max_freq = max(len(word_to_sentences[node]) for node in G.nodes())
    node_sizes = [min(len(word_to_sentences[node]) * max_node_size / max_freq, max_node_size) for node in G.nodes()]

    fig, ax = plt.subplots(figsize=(16, 16))
    nx.draw(G, pos, with_labels=False, node_color=color_map, node_size=node_sizes, width=edge_width, cmap=plt.get_cmap("Pastel2"), ax=ax)

    for node, (x, y) in pos.items():
        plt.text(x, y, node, fontproperties=font, fontsize=16, ha='center', va='center')

    edge_labels = {e: "{:.2f}".format(G[e[0]][e[1]]["weight"]) for e in G.edges}
    for (node1, node2), label in edge_labels.items():
        x1, y1 = pos[node1]
        x2, y2 = pos[node2]
        x, y = (x1 + x2) / 2, (y1 + y2) / 2
        plt.text(x, y, label, fontproperties=font, fontsize=12, ha='center', va='center')
        
    plt.axis('off')
    return fig

# エンコーディングのリスト
encodings = ['shift-jis','utf-8', 'cp932', 'iso-8859-1']

def detect_encoding(file):
    rawdata = file.read()
    result = chardet.detect(rawdata)
    file.seek(0)  # ファイルポインタを最初に戻す
    return result['encoding']

def load_file(uploaded_file):
    """ファイルのエンコーディングを検出し、読み込む関数"""
    encoding = detect_encoding(uploaded_file)
    try:
        # ファイル内容をデコードしてStringIOにラップ
        decoded_data = uploaded_file.read().decode(encoding)
        file_io = io.StringIO(decoded_data)
        df = pd.read_csv(file_io)
        st.sidebar.info(f"ファイルが {encoding} で正常に読み込まれました。")
        return df
    except UnicodeDecodeError as e:
        st.sidebar.error(f"{encoding} での読み込み時にエラーが発生しました: {e}")
        return None

def main():
    st.title('ネコでも使える！会計テキストマイニング：ゼミ版（ネコテキZ）')
    st.write('「ネコでも使える！会計テキストマイニング」（ https://textsan-pxj.streamlit.app/ ）を改良しました。')
    st.write('【改良点】')
    st.write('・CSV形式による複数テキストの入力が可能になりました。')
    st.write('・強制抽出語、排除語、同義語の指定が可能になりました。')
    st.write('・タグと品詞の指定による詳細な分析が可能になりました。')
    st.write('・CSV形式による出力データのダウンロードが可能になりました。')
    
    st.subheader('語の設定')
    # Uploaders
    uploaded_file = st.file_uploader("強制抽出語 CSV ファイルをアップロードしてください", type=['csv'])
    if uploaded_file:
        df_user_dic = pd.read_csv(uploaded_file)
        st.session_state['user_dic_path'] = create_user_dic(df_user_dic)
        st.info("強制抽出語ファイルがアップロードされました。")

    exclusion_file = st.file_uploader("排除語 CSV ファイルをアップロードしてください", type=['csv'])
    if exclusion_file:
        df_exclusion = pd.read_csv(exclusion_file)
        st.session_state['exclusion_words'] = df_exclusion['word'].tolist()
        st.info("排除語ファイルがアップロードされました。")

    synonym_file = st.file_uploader("同義語辞書 CSV ファイルをアップロードしてください", type=['csv'])
    if synonym_file:
        df_synonyms = pd.read_csv(synonym_file)
        st.session_state['synonym_map'] = create_synonym_map(df_synonyms)
        st.info("同義語辞書ファイルがアップロードされました。")

    st.sidebar.subheader('データの入力')
    uploaded_file = st.sidebar.file_uploader("解析したいテキスト CSV ファイルをアップロードしてください", type=['csv'])
    if uploaded_file is not None:
        df_text = load_file(uploaded_file)
        if df_text is not None:
            st.session_state['uploaded_file'] = uploaded_file
            st.session_state['df_text'] = df_text
            st.sidebar.write(df_text.head())  # 例としてデータの先頭を表示
        
    st.subheader('基本分析')
    if st.button("基本分析実行"):
        if 'df_text' in st.session_state:
            df_text = st.session_state['df_text']
            # ここに基本分析のロジックを書く

            basic_analysis_results = []
            all_tokens = []
            for _, row in df_text.iterrows():
                tokens = analyze_text(
                    row['text'],
                    user_dic_path=st.session_state.get('user_dic_path', None),
                    exclusion_words=st.session_state.get('exclusion_words', []),
                    synonym_map=st.session_state.get('synonym_map', {}),
                )
                char_count = len(row['text'])
                word_count = len(tokens)
                sent_count = row['text'].count('。') + row['text'].count('！') + row['text'].count('？') + (1 if row['text'] and row['text'][-1] not in '。！？' else 0)
                unique_words = len(set([token.base_form for token in tokens if token.base_form != '*']))
                basic_analysis_results.append({
                    'tag': row['tag'],
                    'characters': char_count,
                    'words': word_count,
                    'sentences': sent_count,
                    'unique_words': unique_words
                })
                all_tokens.extend(tokens)

            df_basic_analysis_results = pd.DataFrame(basic_analysis_results)
            st.write("基本分析結果（文字数、単語数、文章数、異なり語数）:")
            st.write(df_basic_analysis_results)

            # 結果をCSV形式でダウンロード可能にする
            st.download_button("Download basic analysis Results", df_basic_analysis_results.to_csv(index=False).encode('cp932'), file_name='basic analysis Results.csv')
        
    else:
        st.error('ファイルがアップロードされていません。')

    st.subheader('単語頻度分析')

    # タグの選択
    if 'df_text' in st.session_state:
        df_text = st.session_state['df_text']
        tags = ['all'] + sorted(df_text['tag'].unique())  # 'all'オプションと他のタグを選択肢に追加
        selected_tags = st.multiselect("タグを選択してください", tags, default=['all'], key='tag_select_frequency')

        # 品詞選択のためのユーザーインターフェース
        pos_choice_freq = st.multiselect('品詞を選択してください（単語頻度分析用）', ['名詞', '強制抽出語', '動詞', '形容詞', '副詞', 'すべて'], default=['名詞'], key='pos_select_frequency')

    # 単語頻度分析を実行するボタン
    if st.button("単語頻度分析実行"):
        if 'df_text' in st.session_state and selected_tags:
            df_text = st.session_state['df_text']
            all_tokens = []

            # 選択されたタグに応じてデータをフィルタリング
            if 'all' in selected_tags:
                filtered_texts = df_text
            else:
                filtered_texts = df_text[df_text['tag'].isin(selected_tags)]

            # 全テキストからトークンを収集
            for _, row in filtered_texts.iterrows():
                tokens = analyze_text(
                    row['text'],
                    user_dic_path=st.session_state.get('user_dic_path', None),
                    exclusion_words=st.session_state.get('exclusion_words', []),
                    synonym_map=st.session_state.get('synonym_map', {}),
                    pos_filters=pos_choice_freq
                )
                all_tokens.extend(tokens)

            # 単語の頻度をカウントし、頻度が高い上位*単語を抽出
            word_freq = count_words(all_tokens)
            top_words = pd.DataFrame(list(word_freq.items()), columns=['Word', 'Frequency']).sort_values(by='Frequency', ascending=False).head(100)

            # tagごとに上位30単語の出現頻度をカウント
            tag_word_freq = []
            top_words_set = set(top_words['Word'])  # 上位30単語のセット

            for _, row in filtered_texts.iterrows():
                tokens = analyze_text(
                    row['text'],
                    user_dic_path=st.session_state.get('user_dic_path', None),
                    exclusion_words=st.session_state.get('exclusion_words', []),
                    synonym_map=st.session_state.get('synonym_map', {}),
                    pos_filters=pos_choice_freq
                )
                # このテキストでの上位単語の出現回数をカウント
                current_freq = {word: 0 for word in top_words_set}
                for token in tokens:
                    word = token.base_form if token.base_form != '*' else token.surface
                    if word in current_freq:
                        current_freq[word] += 1

                tag_word_freq.append({'tag': row['tag'], **current_freq})

            # 全体の単語頻度（total）を追加
            total_freq = {word: word_freq[word] for word in top_words_set}
            total_freq['tag'] = 'total'
            tag_word_freq.insert(0, total_freq)  # 全体の頻度をリストの先頭に追加

            # タグごとの上位30単語の出現頻度をDataFrameにして表示
            df_tag_word_freq = pd.DataFrame(tag_word_freq)
            
            # 'total' タグを基にカラム順を並び替え
            sorted_columns = ['tag'] + sorted([col for col in df_tag_word_freq.columns if col != 'tag'], key=lambda x: -df_tag_word_freq.loc[df_tag_word_freq['tag'] == 'total', x].item())
            df_tag_word_freq = df_tag_word_freq[sorted_columns]
            st.write("タグごとの上位100単語の出現頻度:")
            st.write(df_tag_word_freq)
            st.download_button("Download Tag Word Frequency Results", df_tag_word_freq.to_csv(index=False).encode('cp932'), file_name='tag_word_frequency_results.csv')
            
            # 単語の頻度をカウント
            word_freq = count_words(all_tokens)

            # 単語頻度データを元にワードクラウドを生成
            make_wordcloud(word_freq)
            
            st.write("totalによるワードクラウド:")
            st.pyplot(plt)

            # 結果をCSV形式でダウンロード可能にする

        else:
            st.error('データが不足しているか、選択されたタグがありません。')
    else:
        st.error('ファイルがアップロードされていません。')


    st.subheader("共起ネットワークぶんしぇき")
    if 'df_text' in st.session_state:
        df_text = st.session_state['df_text']
        # タグの選択
        tags = ['all'] + list(df_text['tag'].unique())  # 'all'オプションと他のタグを選択肢に追加
        selected_tags = st.multiselect("タグを選択してください", tags, default=['all'], key='tag_select_network')

    # 品詞の選択
    pos_options = ['名詞', '強制抽出語', '動詞', '形容詞', '副詞', 'すべて']  # 使用可能な品詞オプション
    selected_pos = st.multiselect("品詞を選択してください（共起ネットワーク用）", pos_options, default=['名詞'], key='pos_select_network')

    jaccard_threshold = st.slider("Jaccard係数の閾値", min_value=0.0, max_value=1.0, value=0.2, step=0.05, key="jaccard_threshold")
    min_word_freq = st.slider("単語の最小出現数", min_value=1, max_value=50, value=5, key="min_word_freq")
    min_cooccurrence = st.slider("最小共起数", min_value=1, max_value=10, value=2, key="min_cooccurrence")
    top_n_edges = st.slider("上位共起関係の数", min_value=1, max_value=60, value=30, key="top_n_edges")    

    if 'df_text' in st.session_state:
        df_text = st.session_state['df_text']
        # ここに共起ネットワーク分析のロジックを書く

        if st.button("共起ネットワークを表示"):
            if 'df_text' in st.session_state:
                df_text = st.session_state['df_text']
                
                if 'all' in selected_tags or len(selected_tags) == 0:
                    texts = df_text['text'].tolist()  # 直接テキストのリストを抽出
                else:
                    texts = df_text[df_text['tag'].isin(selected_tags)]['text'].tolist()

                # make_network_with_jaccard関数の呼び出しを更新
                network = make_network_with_jaccard(texts, jaccard_threshold, min_word_freq, min_cooccurrence, top_n_edges, selected_pos, st.session_state.get('user_dic_path', None), st.session_state.get('exclusion_words', []), st.session_state.get('synonym_map', {}))

                if not network:
                    st.error("指定された条件で単語の共起ネットワークが見つかりませんでした。")
                else:
                    st.pyplot(network)
            else:
                st.error("ファイルがアップロードされていません。")

        else:
            st.error("ファイルがアップロードされていません。")
            
    st.subheader("【サイト運営者】")
    st.write("青山学院大学　経営学部　矢澤憲一研究室")
    st.subheader("【諸注意】")
    st.write("１．私的目的での利用について：")
    st.write("本ウェブアプリケーションは、個人的な用途で自由にご利用いただけます。しかしながら、公序良俗に反する行為は固く禁じられています。利用者の皆様には、社会的な規範を尊重し、責任ある行動をお願いいたします。")
    st.write("２．ビジネス目的での利用について：")
    st.write("本アプリケーションをビジネス目的で使用される場合は、事前に以下の連絡先までご一報ください。使用に関する詳細な情報を提供いたします。")
    st.write("２．学術論文執筆目的での利用について：")
    st.write("学術論文の執筆に当たり、本アプリケーションのデータや機能を利用される場合は、下記の文献を参考文献として明記し、同時に以下の連絡先までご一報いただくようお願いいたします。")
    st.write("参考文献：執筆中")
    st.write("連絡先：yazawa(at)busi.aoyama.ac.jp")
    st.subheader("【免責事項】")
    st.write("このウェブサイトおよびそのコンテンツは、一般的な情報提供を目的としています。このウェブサイトの情報を使用または適用することによって生じるいかなる利益、損失、損害について、当ウェブサイトおよびその運営者は一切の責任を負いません。情報の正確性、完全性、時宜性、適切性についても、一切保証するものではありません。")



if __name__ == '__main__':
    main()

#%%
