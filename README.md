# 🏋️‍♂️ Bicep Curl Counter Program

## 📋 Deskripsi
Program ini menggunakan **MediaPipe** dan **Machine Learning** untuk mendeteksi gerakan angkat beban berdasarkan sudut siku kanan dan kiri. Program menghitung jumlah pengangkatan beban secara terpisah untuk tangan kanan dan kiri, serta menampilkan total pengangkatan. Program berjalan secara real-time menggunakan webcam.

---

## 🛠️ Fitur Utama
- **Deteksi Gerakan Angkat Beban**: Mendeteksi gerakan berdasarkan sudut siku kanan dan kiri.
- **Perhitungan Terpisah**: Menghitung jumlah pengangkatan untuk tangan kanan dan kiri secara independen.
- **Total Pengangkatan**: Menampilkan total jumlah pengangkatan dari kedua tangan.
- **Real-Time Feedback**: Menampilkan sudut siku kanan dan kiri serta status pengangkatan langsung di layar.

---

## 📦 Persyaratan Sistem
- Python >= 3.7
- Webcam
- Library Python:
  - `opencv-python`
  - `mediapipe`
  - `numpy`
  - `scikit-learn`

---

## ⚙️ Instalasi
1. **Clone Repository**
   ```bash
   git clone https://github.com/mochshultan/BicepCurl_Counter.git
   cd BicepCurl_Counter
   ```
2. **Install Kaggle Data Set**
   ```
   import kagglehub
   path = kagglehub.dataset_download("trainingdatapro/pose-estimation")
   print("Path to dataset files:", path)
   ```
   Note: Adjust path on `main.py` due to where you put your dataset
3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run Program**
   ```bash
   python main.py
   ```

---

## 🎮 Cara Menggunakan
1. Pastikan webcam terhubung dan berfungsi.
2. Berdiri di depan webcam dengan bahu, siku, dan pergelangan tangan terlihat jelas.
3. Angkat beban dengan tangan kanan, kiri, atau keduanya. Program akan mendeteksi dan menghitung pengangkatan.
4. Tekan tombol `r` pada keyboard untuk reset value counter
5. Tekan tombol `q` pada keyboard untuk keluar dari program.

---

## 🖥️ Tampilan Program
- **Sudut Siku**: Ditampilkan secara real-time.
- **Jumlah Pengangkatan**: Untuk tangan kanan, kiri, dan total.
- **Status**: "lifting" atau "not lifting" berdasarkan gerakan.

---

## 📊 Logika Deteksi
- **Sudut Siku**: Jika sudut < threshold, dianggap sebagai angkat beban (using 55°).
- **Decision Tree**: Model dilatih dengan memasukkan data training berupa pasangan sudut siku (contoh: sudut siku kanan dan kiri) serta label yang merepresentasikan suatu kondisi (misalnya, aksi angkat beban atau tidak). Decision Tree mempelajari threshold atau batas nilai sudut yang memisahkan antara kondisi tersebut dengan membangun struktur pohon "if‑then". Saat proses pelatihan (dengan metode fit), algoritma mencari aturan terbaik untuk membagi data sehingga tiap cabang dari pohon menghasilkan prediksi yang tepat

---

## 📁 Struktur Program
- `main.py` : File utama program.
- `utils.py` : Fungsi `calculate_angle` untuk menghitung sudut siku dari koordinat landmark.

---

## 🧪 Pengembangan & Testing
- **Pengembangan**: Tambahkan data pelatihan untuk meningkatkan akurasi atau gunakan model lebih kompleks seperti TensorFlow.
- **Testing**: Uji dengan berbagai posisi tubuh dan sudut kamera.

---

## 🚀 Fitur Pengembangan Selanjutnya
- Deteksi apakah pengguna benar-benar memegang beban.
- Hitung estimasi kalori yang terbakar.
- Simpan data pengangkatan ke file/database.
- Tambahkan antarmuka GUI.

---

## ❓ FAQ
- **Program tidak mendeteksi gerakan?**
  - Pastikan bahu, siku, dan pergelangan tangan terlihat jelas oleh kamera.
  - Pastikan pencahayaan cukup.
- **Bagaimana meningkatkan akurasi deteksi?**
  - Tambahkan data pelatihan pada model decision tree.
  - Gunakan kamera resolusi tinggi.
- **Apakah bisa digunakan tanpa beban?**
  - Ya, deteksi dilakukan pada berdasarkan sudut siku, bukan keberadaan beban (sistem sederhana).

---

## 📞 Kontak
- **Email**: moch.ultan.ali-2023@ftmm.unair.ac.id
- **GitHub**: [https://github.com/mochshultan](https://github.com/mochshultan)

---

## 📜 Lisensi
Program ini dilisensikan di bawah MIT License. Bebas digunakan, dimodifikasi, dan didistribusikan dengan atribusi kepada pembuat.
