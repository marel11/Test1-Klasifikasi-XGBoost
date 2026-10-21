# Pengumpulan Test 1 - Klasifikasi XGBoost

**Nama:** Mochammad Maril Livio  \n**NIM:** 231230065  \n**Universitas:** Universitas Muhammadiyah Pontianak - Sistem Informasi

## Lima item yang dikumpulkan

1. `MochammadMarilLivio_Test1_KlasifikasiXGBoost.doc` - ringkasan pengujian dan grafik.
2. `MochammadMarilLivio_Test1_KlasifikasiXGBoost.csv` - hasil normalisasi dan encoding, 1.826 baris.
3. `MochammadMarilLivio_Test1_KlasifikasiXGBoost.py` - kode analisis, pelatihan, evaluasi, grafik dan ekspor normalisasi.
4. **Visualisasi GitHub:** upload seluruh isi folder, lalu gunakan halaman `HASIL_VISUALISASI.md` sebagai link visualisasi; tempel URL yang benar pada dua dokumen laporan.
5. `MochammadMarilLivio_Laporan Praktikum 1.doc` - laporan lengkap dengan delapan gambar beserta analisisnya.

Dua file `.docx` juga disertakan sebagai format yang mudah diedit sebelum diserahkan.

## Cara menjalankan (MacBook/VS Code)

Pastikan Terminal menggunakan Conda environment tempat semua paket terinstal. Contoh environment yang bersih:

```bash
conda create -n xgboost_bersih -c conda-forge --override-channels --strict-channel-priority python=3.11 xgboost llvm-openmp streamlit pyarrow pandas numpy scikit-learn matplotlib seaborn plotly joblib -y
conda activate xgboost_bersih
python MochammadMarilLivio_Test1_KlasifikasiXGBoost.py
python -m streamlit run app.py
```

Buka `http://localhost:8501` saat Streamlit berjalan. Jika Conda sudah aktif dengan environment yang sehat, tidak perlu membuat ulang environment.

## Isi folder

- `data/`: CSV mentah dari pengguna (tidak dinormalisasi, tidak diubah).
- `MochammadMarilLivio_Test1_KlasifikasiXGBoost.csv`: dataset yang dinormalisasi menggunakan MinMaxScaler dengan fit hanya pada training, plus encoding.
- `figures/`: delapan grafik, termasuk confusion matrix, ROC Curve, dan feature importance.
- `outputs/`: model `joblib`, metrik JSON, prediksi data uji, ringkasan korelasi.
- `app.py`: dashboard Streamlit interaktif.
- `analysis_pipeline.py`: salinan kode utama agar `app.py` dapat mengimpor fungsi pemrosesan.
- `HASIL_VISUALISASI.md`: markdown untuk menampilkan grafik di GitHub.

## Ringkasan hasil aktual

| Evaluasi | Hasil |
|---|---:|
| Accuracy | 90.44% |
| Precision | 90.23% |
| Recall | 93.27% |
| F1-Score | 91.73% |
| ROC-AUC | 0.9604 |

Confusion matrix: TN=137, FP=21, FN=14, TP=194.

## Menambahkan link GitHub pada laporan

1. Buat repository GitHub baru, misalnya `praktikum-xgboost-kolkata`.
2. Unggah folder proyek: semua file utama dan `figures/`, `data/`, `outputs/`.
3. Buka file `HASIL_VISUALISASI.md` yang sudah tampil dengan gambar.
4. Salin URL repository dan URL halaman `HASIL_VISUALISASI.md`.
5. Buka dokumen `.docx` atau `.doc` dan ganti dua placeholder `[ISI URL ...]` pada bagian GitHub, kemudian simpan.

**Penting:** Link GitHub belum dapat diisi otomatis karena repository pengguna belum diunggah. GitHub menampilkan grafik statis; untuk situs dashboard interaktif perlu deploy Streamlit secara terpisah.

## Catatan data

Berkas berjudul `jan_june_2026` ternyata memuat tanggal 30 Juni 2021 s.d. 30 Juni 2026. Metadata sumber CSV asli tidak disertakan; jangan mengklaim asal unduhan tertentu.
