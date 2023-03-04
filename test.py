import streamlit as st
import nltk
from nltk.util import ngrams
from collections import Counter
import collections
import re
import math

# Structure of the web
header=st.container()
q1 = st.container()
q2 = st.container()
q3 = st.container()

#Introduction about the program
with header:
    st.markdown("<h1 style='text-align: center; color: lightblue;'>Bài thi đánh giá năng lực – Thực tập sinh Xử lý ngôn ngữ tự nhiên</h1>", unsafe_allow_html=True)
    st.caption("<h1 style='text-align: center;'>Thí sinh Bùi Tuấn Nghĩa</h1>",unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: left; color: red;'>Tổng quan về format bài dự thi</h1>", unsafe_allow_html=True)
    st.markdown("""Em xin phép được nộp bài thi dưới dạng website và tất cả các bài thi liên quan đến code, em sẽ để full code tại web và cũng như thực hiện việc chạy code
                dựa trên các văn bản tự nhập hoặc dữ liệu em tự thu thập tùy câu hỏi. Những câu hỏi liên quan đến lý thuyết em sẽ viết cách diễn đạt bằng Tiếng Việt ạ. Ngoài ra 
                em cũng sẽ đính kèm link Github nếu như các anh chị muốn xem qua toàn bộ code cho bài thi cũng như website ạ.""")
    st.write("---")

# Question 1
with q1:
    st.markdown("<h1 style='text-align: left; color: red;'>01: N-gram</h1>", unsafe_allow_html=True)
    st.subheader('a,')
    st.write('- Theo em được biết, n-gram trong xử lý ngôn ngữ tự nhiên là  tần suất xuất hiện của n kí tự (từ) liên tiếp xuất hiện trong dữ liệu.')
    st.write("- 1 số mô hình n-gram phổ biến bao gồm:")
    st.write(" Unigram, mô hình với n = 1. Ví dụ: ['Bùi', 'Tuấn', 'Nghĩa'] ")
    st.write(" Bigram, mô hình với n = 2. Ví dụ: ['Bùi Tuấn', 'Nghĩa lớp', '12 Tin'] ")
    st.write(" Trigram, mô hình với n = 3. Ví dụ: ['Bùi Tuấn Nghĩa', 'lớp 12 Tin'] ")
    
    st.subheader('b,')
    
    # Input sentence
    paragraph = st.text_area('Hãy nhập tập các câu mà mọi người chọn ạ (Monolingual corpus only).')

    if paragraph != ' ':
        sentences = paragraph.split('.')

        # Initialize lists to store n-grams for all sentences
        all_two_grams = []
        all_three_grams = []

        # Iterate over each sentence
        for sentence in sentences:
            # Tokenize the sentence
            tokens = nltk.word_tokenize(sentence)
            
            # Generate 2-gram and 3-gram for this sentence
            two_grams = list(ngrams(tokens, 2))
            three_grams = list(ngrams(tokens, 3))
            
            # Add the n-grams for this sentence to the overall list
            all_two_grams.extend(two_grams)
            all_three_grams.extend(three_grams)

        # Count the occurrences of each 2-gram and 3-gram
        two_gram_counts = Counter(all_two_grams)
        three_gram_counts = Counter(all_three_grams)

        # Calculate the probabilities of each 2-gram and 3-gram
        total_two_gram_count = sum(two_gram_counts.values())
        total_three_gram_count = sum(three_gram_counts.values())

        two_gram_probabilities = {gram: count/total_two_gram_count for gram, count in two_gram_counts.items()}
        three_gram_probabilities = {gram: count/total_three_gram_count for gram, count in three_gram_counts.items()}

        # Print the 2-gram and 3-gram and their probabilities in 2 different sides
        left, right = st.columns(2)
        with left:
            st.write("2-gram và xác suất tương ứng")
            for gram, prob in two_gram_probabilities.items():
                st.write(gram, prob)
        
        with right:
            st.write("3-gram và xác suất tương ứng")
            for gram, prob in three_gram_probabilities.items():
                st.write(gram, prob)

    # Review code section
    view_code = st.checkbox('Source code cho câu b:')
    if view_code:
        st.code("""paragraph = st.text_area('Hãy nhập tập các câu mà mọi người chọn ạ (Monolingual corpus only).')

if paragraph != ' ':
    sentences = paragraph.split('.')

    # Initialize lists to store n-grams for all sentences
    all_two_grams = []
    all_three_grams = []

    # Iterate over each sentence
    for sentence in sentences:
        # Tokenize the sentence
        tokens = nltk.word_tokenize(sentence)
        
        # Generate 2-gram and 3-gram for this sentence
        two_grams = list(ngrams(tokens, 2))
        three_grams = list(ngrams(tokens, 3))
        
        # Add the n-grams for this sentence to the overall list
        all_two_grams.extend(two_grams)
        all_three_grams.extend(three_grams)

    # Count the occurrences of each 2-gram and 3-gram
    two_gram_counts = Counter(all_two_grams)
    three_gram_counts = Counter(all_three_grams)

    # Calculate the probabilities of each 2-gram and 3-gram
    total_two_gram_count = sum(two_gram_counts.values())
    total_three_gram_count = sum(three_gram_counts.values())

    two_gram_probabilities = {gram: count/total_two_gram_count for gram, count in two_gram_counts.items()}
    three_gram_probabilities = {gram: count/total_three_gram_count for gram, count in three_gram_counts.items()}

    # Print the 2-gram and 3-gram and their probabilities
    left, right = st.columns(2)
    with left:
        st.write("2-gram và xác suất tương ứng")
        for gram, prob in two_gram_probabilities.items():
            st.write(gram, prob)

    with right:
        st.write("3-gram và xác suất tương ứng")
        for gram, prob in three_gram_probabilities.items():
            st.write(gram, prob) """)
    
    st.subheader('c, 3-gram và xác suất tương ứng khi dùng Jelinek-Mercer smoothing :')
    st.write("""Jelinek-Mercer smoothing hay còn gọi là Interpolation smoothing là 1 phương pháp được dùng để làm tính toán phân bố xác suất của n-gram dựa trên 1 corplus cho trước.
                Thuật toán của Jelinek-Mercer được tính theo maximum-likelihood language model trên tần số xuất hiện của từ tương ứng và xác suất của n-gram nhỏ hơn. Ví dụ với 3-gram thì
                sẽ tính xác suất của 1-gram và 2 gram và cũng tùy thuộc vào giá trị λ dựa trên model muốn xử lý. Gía trị trung bình của λ là 0.5""")
    
    #jelinek_mercer_smoothing probability Implementation: auto set λ = 0.5
    def jelinek_mercer_smoothing(unigrams, bigrams, trigrams, alpha=0.5):
        trigram_probabilities = {}
        
        # Sum of the triagram values in corplus
        total_trigram_count = sum(trigrams.values())

        for trigram, count in trigrams.items():
            word1, word2, word3 = trigram
            
            # Get unigram and bigram for calculation
            bigram_count = bigrams.get((word1, word2), 0)
            unigram_count = unigrams.get(word3, 0)
            
            #Count probability based on formula
            probability = (1 - alpha) * (count / total_trigram_count)
            probability += alpha * (1 - alpha) * (bigram_count / sum(bigrams.values()))
            probability += alpha * (unigram_count / sum(unigrams.values()))
            trigram_probabilities[trigram] = probability

        return trigram_probabilities

    # Tokenize the sentence
    tokens = [sentence.split() for sentence in sentences]
    unigrams = collections.Counter([token for sentence in tokens for token in sentence])
    bigrams = collections.Counter([(tokens[i][j], tokens[i][j+1]) for i in range(len(tokens)) for j in range(len(tokens[i])-1)])
    trigrams = collections.Counter([(tokens[i][j], tokens[i][j+1], tokens[i][j+2]) for i in range(len(tokens)) for j in range(len(tokens[i])-2)])

    # Using the jelinek_mercer_smoothing function to count probability
    smoothed_trigram_probabilities = jelinek_mercer_smoothing(unigrams, bigrams, trigrams)

    #Print out the result
    for trigram, probability in smoothed_trigram_probabilities.items():
        st.write(trigram, probability)
    
    # Review code section
    view_code1 = st.checkbox('Source code cho câu c:')
    if view_code1:
        st.code("""def jelinek_mercer_smoothing(unigrams, bigrams, trigrams, alpha=0.5):
        trigram_probabilities = {}
        total_trigram_count = sum(trigrams.values())

        for trigram, count in trigrams.items():
            word1, word2, word3 = trigram
            bigram_count = bigrams.get((word1, word2), 0)
            unigram_count = unigrams.get(word3, 0)
            probability = (1 - alpha) * (count / total_trigram_count)
            probability += alpha * (1 - alpha) * (bigram_count / sum(bigrams.values()))
            probability += alpha * (unigram_count / sum(unigrams.values()))
            trigram_probabilities[trigram] = probability

        return trigram_probabilities

tokens = [sentence.split() for sentence in sentences]
unigrams = collections.Counter([token for sentence in tokens for token in sentence])
bigrams = collections.Counter([(tokens[i][j], tokens[i][j+1]) for i in range(len(tokens)) for j in range(len(tokens[i])-1)])
trigrams = collections.Counter([(tokens[i][j], tokens[i][j+1], tokens[i][j+2]) for i in range(len(tokens)) for j in range(len(tokens[i])-2)])

smoothed_trigram_probabilities = jelinek_mercer_smoothing(unigrams, bigrams, trigrams)

for trigram, probability in smoothed_trigram_probabilities.items():
    st.write(trigram, probability) """)

with q2:
    st.markdown("<h1 style='text-align: left; color: red;'>02: Biểu thức chính quy (Regex)</h1>", unsafe_allow_html=True)
    paragraph1 = st.text_area('Hãy nhập văn bản mà mọi người muốn trích xuất được số điện thoại di động ở Việt Nam.')
    st.subheader('a,')
    # Vietnamese Phone number regex in the form of 09xxxxxxxx, 03xxxxxxxx, 05xxxxxxxx, 07xxxxxxxx, 08xxxxxxxx.
    phone = re.findall(r'\b(09\d{8}|03\d{8}|05\d{8}|07\d{8}|08\d{8})\b', paragraph1)
    st.write("Kết quả là:")
    st.write(phone)
    
    # Review code section
    view_code = st.checkbox('Source code cho câu 2a:')
    if view_code:
        st.code("matches = re.findall(r'\b(09\d{8}|03\d{8}|05\d{8}|07\d{8}|08\d{8})\b', paragraph1)")

    st.subheader('b,')
    st.write("Kết quả tìm url là:")
    urls = re.findall('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', paragraph1)
    st.write(urls)
    
    # Review code section
    view_code = st.checkbox('Source code cho câu 2b phần url:')
    if view_code:
        st.code("urls = re.findall('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', paragraph1)")
        
    st.write("Kết quả tìm email là:")
    emails = re.findall(r'[\w\.-]+@[\w\.-]+\.[\w\.-]+', paragraph1)
    st.write(emails)
    
    # Review code section
    view_code = st.checkbox('Source code cho câu 2b phần email:')
    if view_code:
        st.code("emails = re.findall(r'[\w\.-]+@[\w\.-]+\.[\w\.-]+', paragraph1)")

with q3:
    st.markdown("<h1 style='text-align: left; color: red;'>03: Kiến thức chung về Machine Learning</h1>", unsafe_allow_html=True)
    sentence = st.text_area('Hãy nhập 1 đoạn văn:')

    # Define the documents
    docs = []
    docs = sentence.split('.')

    if len(docs) > 1:
        # Split the documents into words
        word_lists = [doc.split() for doc in docs]

        # Get the unique words in all the documents
        word_set = set().union(*word_lists)

        # Create a dictionary with all words and initialize the count to zero
        word_counts = [dict.fromkeys(word_set, 0) for i in range(len(docs))]

        # Count the occurrence of each word in each document
        for i, word_list in enumerate(word_lists):
            for word in word_list:
                word_counts[i][word] += 1

        # Compute the term frequency (TF) for each word in each document
        tf = [dict((word, count/float(len(word_list))) if len(word_list) > 0 else (word, 0) for word, count in word_count.items()) for word_list, word_count in zip(word_lists, word_counts)]

        # Compute the inverse document frequency (IDF) for each word
        idf = dict((word, math.log(len(docs)/sum(1 for doc in docs if word in doc))) for word in word_set)

        # Compute the TF-IDF score for each word in each document
        tf_idf = []
        for word_count, tf_doc in zip(word_counts, tf):
            doc_tfidf = {}
            for word, value in tf_doc.items():
                doc_tfidf[word] = value*idf[word]
            tf_idf.append(doc_tfidf)
            
        st.subheader('a,')
        st.write("Kết quả là:")
        st.write(tf_idf)
        
        # Review code section
        view_code = st.checkbox('Source code cho câu 3a:')
        if view_code:
            st.code("""sentence = st.text_area('Hãy nhập 1 đoạn văn:')

# Define the documents
docs = []
docs = sentence.split('.')

if len(docs) > 1:
    # Split the documents into words
    word_lists = [doc.split() for doc in docs]

    # Get the unique words in all the documents
    word_set = set().union(*word_lists)

    # Create a dictionary with all words and initialize the count to zero
    word_counts = [dict.fromkeys(word_set, 0) for i in range(len(docs))]

    # Count the occurrence of each word in each document
    for i, word_list in enumerate(word_lists):
        for word in word_list:
            word_counts[i][word] += 1

    # Compute the term frequency (TF) for each word in each document
    tf = [dict((word, count/float(len(word_list))) if len(word_list) > 0 else (word, 0) for word, count in word_count.items()) for word_list, word_count in zip(word_lists, word_counts)]


    # Compute the inverse document frequency (IDF) for each word
    idf = dict((word, math.log(len(docs)/sum(1 for doc in docs if word in doc))) for word in word_set)

    # Compute the TF-IDF score for each word in each document
    tf_idf = []
    for word_count, tf_doc in zip(word_counts, tf):
        doc_tfidf = {}
        for word, value in tf_doc.items():
            doc_tfidf[word] = value*idf[word]
        tf_idf.append(doc_tfidf)
        
    st.subheader('a,')
    st.write("Kết quả là:")
    st.write(tf_idf) """)
        
    else:
        st.warning("""Em cũng không biết rõ là mình đúng hay không nhưng theo em tìm hiểu thì tf_idf được áp dụng với bag_of word sẽ được xử lý trong trường hợp văn bản có nhiều hơn 1 câu.
                   Nếu chỉ có 1 câu thì tất cả giá trị tf của câu 1 sẽ được auto assume là 0 và kết quả sẽ chỉ là 1 dictionary với các word đi kèm giá trị 0.""")
    
    st.subheader('c')
    st.write("Để có thể đánh giá chất lượng của 1 mô hình phân loại, người ta thường dùng các độ đo như Classification Accuracy, Precision, Recall, F1-Score, Sensitivity- Specificity và AUC.")
    st.write("- Classification Accuracy: đánh giá xem tỷ lệ đoán chính xác là bao nhiêu phần trăm nhưng không chỉ ra cụ thể mỗi loại được phân loại như thế nào.")
    st.write("- Precision: đánh giá độ chính xác của model khi dự đoán từng mục nhất định, sẽ đánh giá độ cụ thể tốt hơn Classification Model là dự đoán overall. ")
    st.write("- Recall: đánh giá tỷ lệ của dự đoán đúng với tất cả những trường hợp mà có khả năng đúng.")
    st.write("- F1-Score: kết hợp giữa Recall và Precision")
    st.write("- Sensitivity-Specificity: được sử dụng chủ yếu trong y tế và sinh học, công thức của Sensitivity sẽ giống Recall và Specificity sẽ ngược lại với Sensitivity.")
    st.write("- AUC: em không rõ về thang đo này lắm ạ. ")
    
    
    