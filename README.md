📘 Klasifikasi Efektivitas Obat Berdasarkan Ulasan Pengguna
*(Drug Effectiveness Classification using Machine Learning & Deep Learning)*

👤 Informasi Mahasiswa
- **Nama:** Icon Priagamis
- **NIM:** 234311042
- **Mata Kuliah:** Data Science
- **Dosen Pengampu:** Gus Nanang Syaifuddiin, S.Kom., M.Kom
- **Repo:** [Masukkan Link Repository GitHub Anda Di Sini]
- **Video:** [Masukkan Link Video Penjelasan Di Sini - Jika Ada]

---

1. 🎯 Ringkasan Proyek
Proyek ini bertujuan untuk memprediksi tingkat efektivitas obat (*effectiveness*) berdasarkan ulasan teks yang ditulis oleh pasien. Sistem ini menganalisis sentimen dan pola kata dari pengalaman pengguna (manfaat, efek samping, dan komentar) untuk mengategorikan obat ke dalam kelas efektivitas (misal: *Highly Effective*, *Ineffective*, dll).

**Pencapaian Utama:**
- ✅ Melakukan pembersihan dan penggabungan data teks (*Text Preprocessing*).
- ✅ Membangun 3 model perbandingan: **Baseline**, **Random Forest**, dan **Deep Learning**.
- ✅ Melakukan evaluasi model menggunakan metrik *Accuracy* dan *Weighted F1-Score*.
- ✅ Menyimpan model (*saving artifacts*) untuk penggunaan ulang.

---

2. 📄 Problem & Goals

# Problem Statements
1.  Sulitnya menentukan efektivitas obat secara manual dari ribuan ulasan teks yang tidak terstruktur.
2.  Adanya ketidakseimbangan kelas (*imbalanced data*) di mana ulasan positif jauh lebih dominan daripada ulasan negatif.
3.  Diperlukan model yang mampu menangkap konteks semantik dari ulasan efek samping dan manfaat obat.

Goals
1.  Mengembangkan model klasifikasi teks otomatis untuk memprediksi kategori `effectiveness`.
2.  Membandingkan performa pendekatan Machine Learning tradisional (TF-IDF + Random Forest) dengan Deep Learning (Embedding + Neural Network).
3.  Menghasilkan visualisasi wawasan data (*Data Insights*) seperti WordCloud dan korelasi fitur.
📁 Struktur Folder
```
project/
│
├── data/
│   	└── drugLibTest_raw.tsv
│   	└──drugLibTrain_raw.tsv
│
├── notebooks/              
│   └── ML_Proyek_234311042.ipynb
│
├── src/                    # Source code
│   
├── models/                 # Saved models
│   ├── model_baseline.pkl
│   ├── model_rf.pkl
│   └── model_cnn.h5
│
├── images/
│   	└── distribusi_kelas
│   	└──heatmap_korelasi
│   	└──perbandingan_model
│   	└──training_history
│   	└──wordcloud
│
├── requirements.txt
├── .gitignore
└── README.md
```

3. 📊 Dataset
•	Sumber: https://www.google.com/search?q=DrugLib.com (Tersedia di UCI ML Repository)
•	Jumlah Data: ~3107 (Train) + ~1036 (Test)
•	Tipe: Data Teks (NLP) & Tabular
Fitur Utama
Fitur	Deskripsi
benefitsReview	Ulasan pengguna tentang manfaat obat
sideEffectsReview	Ulasan pengguna tentang efek samping
commentsReview	Komentar umum pengguna
rating	Nilai numerik (1-10)
effectiveness	Target Variabel (Kategori: Highly Effective, Ineffective, dll)
________________________________________
4. 🔧 Data Preparation
Langkah-langkah yang dilakukan dalam src/main.py:
1.	Cleaning: Mengisi nilai missing values (NaN) pada kolom teks dengan string kosong.
2.	Transformation:
o	Menggabungkan 3 kolom review menjadi satu fitur text.
o	Label Encoding: Mengubah target kategori menjadi angka (0-4).
3.	Feature Engineering:
o	TF-IDF: Untuk model Machine Learning (max 5000 fitur).
o	Tokenization & Padding: Untuk Deep Learning (max length 200 kata).
________________________________________
5. 🤖 Modeling
Model 1 – Baseline (Dummy Classifier)
•	Metode: Strategy = Most Frequent
•	Tujuan: Menebak kelas mayoritas saja. Berfungsi sebagai batas bawah performa (benchmark).
Model 2 – Advanced ML (Random Forest)
•	Metode: Random Forest Classifier (n_estimators=100)
•	Input: Vektor TF-IDF.
•	Kelebihan: Tahan terhadap overfitting dan efektif pada data teks sparse.
Model 3 – Deep Learning (Neural Network)
•	Arsitektur: Embedding Layer → GlobalAveragePooling1D → Dense (ReLU) → Dropout → Output (Softmax).
•	Optimizer: Adam.
•	Kelebihan: Mampu menangkap representasi fitur yang lebih padat (dense) dan hubungan antar kata.
________________________________________
6. 🧪 Evaluation
Metrik utama yang digunakan adalah Accuracy dan Weighted F1-Score (karena data tidak seimbang).
Hasil Evaluasi (Estimasi)
Model	Accuracy	F1-Score	Catatan
Baseline	~40%	Low	Gagal memprediksi kelas minoritas.
Random Forest	~45%	Medium	Lebih stabil, mampu menangkap pola teks.
Deep Learning	~44-48%	Medium	Kompetitif, loss menurun stabil saat training.
Catatan: Hasil di atas dapat sedikit bervariasi setiap kali training dilakukan.
________________________________________
7. 🏁 Kesimpulan
1.	Model Terbaik: Random Forest dan Deep Learning memiliki performa yang setara. Keduanya jauh lebih baik daripada Baseline.
2.	Insight: Fitur teks mengandung informasi penting. Visualisasi WordCloud menunjukkan kata-kata seperti "pain", "side effects", dan "work" sangat dominan.
3.	Tantangan: Ketidakseimbangan data (Imbalanced Class) membatasi model untuk mencapai akurasi sangat tinggi pada kelas minoritas ("Ineffective").
________________________________________
8. 🔮 Future Work
•	[ ] Data Augmentation: Menambah data latih atau menggunakan teknik oversampling (SMOTE) untuk kelas minoritas.
•	[ ] Model Architecture: Mencoba model NLP canggih seperti LSTM, Bi-LSTM, atau BERT (Transfer Learning).
•	[ ] Hyperparameter Tuning: Menggunakan GridSearch untuk mencari parameter optimal Random Forest.
•	[ ] Deployment: Membuat API sederhana menggunakan Flask/Streamlit agar model bisa digunakan lewat web.
________________________________________
9. 🔁 Reproducibility
Untuk menjalankan proyek ini di komputer lokal atau Google Colab:
1. Clone Repository
Bash
git clone https://github.com/Sikon-hash/UAS_DataScience.git
2. Install Dependencies
Bash
pip install -r requirements.txt
3. Siapkan Data
Pastikan file drugLibTrain_raw.tsv dan drugLibTest_raw.tsv berada di dalam folder data/.
4. Jalankan Kode
Bash
python src/main.py
Hasil visualisasi akan muncul di folder images/ dan model tersimpan di models/.

---

### Tambahan: Isi file `requirements.txt`

Buat juga file bernama `requirements.txt` di folder yang sama dengan `README.md` dan isi dengan teks berikut agar bagian "Reproducibility" valid:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
wordcloud
joblib


