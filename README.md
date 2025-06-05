# Laporan-Proyek-ML-Dion-Febri-Setiawan

## Domain Proyek: Prediksi Biaya Asuransi

Asuransi kesehatan merupakan salah satu aspek penting dalam sistem perlindungan finansial masyarakat terhadap risiko kesehatan. Perusahaan asuransi menetapkan biaya premi berdasarkan sejumlah faktor yang merepresentasikan risiko kesehatan calon pemegang polis, seperti usia, status merokok, jenis kelamin, dan indeks massa tubuh (BMI). Penentuan biaya yang tidak akurat dapat menyebabkan ketidakseimbangan antara risiko dan premi, baik merugikan perusahaan maupun pelanggan.

Dengan perkembangan teknologi dan ketersediaan data historis, machine learning dapat dimanfaatkan untuk memprediksi biaya asuransi secara lebih akurat dan efisien. Model prediktif ini dapat membantu perusahaan asuransi membuat keputusan berbasis data dalam menetapkan premi yang adil dan tepat sasaran.

Proyek ini berfokus pada pembangunan model prediksi biaya asuransi individu berdasarkan data demografis dan gaya hidup, seperti usia, jenis kelamin, status merokok, BMI, dan jumlah tanggungan. Model ini diharapkan dapat meningkatkan akurasi estimasi biaya dan mendukung proses underwriting secara otomatis.

## Business Understanding

Dalam industri asuransi kesehatan, perusahaan perlu menentukan biaya premi yang sesuai berdasarkan risiko masing-masing individu. Jika premi terlalu rendah untuk individu berisiko tinggi, perusahaan bisa mengalami kerugian. Sebaliknya, jika premi terlalu tinggi untuk individu berisiko rendah, pelanggan mungkin merasa tidak adil dan beralih ke penyedia lain. Oleh karena itu, pemodelan prediktif berbasis machine learning dapat menjadi solusi yang efisien untuk memperkirakan biaya asuransi berdasarkan faktor-faktor yang relevan.

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Bagaimana cara memanfaatkan data demografis dan gaya hidup (usia, jenis kelamin, BMI, status merokok, jumlah anak, dan wilayah) untuk memprediksi biaya asuransi kesehatan?
- Algoritma machine learning mana yang paling sesuai untuk membangun model prediksi biaya asuransi dengan akurasi yang baik?
- Sejauh mana pengaruh setiap variabel input terhadap besarnya biaya asuransi yang diprediksi?

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Mengembangkan model machine learning untuk memprediksi biaya asuransi berdasarkan data individu.
- Mengevaluasi dan membandingkan kinerja beberapa algoritma regresi seperti Linear Regression, Decision Tree, dan Random Forest.
- Menentukan fitur-fitur yang paling berpengaruh dalam menentukan besarnya premi atau biaya asuransi.

**Rubrik/Kriteria Tambahan (Opsional)**:

    Solution statements
    - Mengimplementasikan minimal dua algoritma regresi (misalnya, Linear Regression, Random Forest Regressor, dan Gradient Boosting) dan membandingkan performanya menggunakan metrik evaluasi seperti RMSE, MAE, dan R² Score.

    - Melakukan hyperparameter tuning pada model terbaik untuk meningkatkan performa prediksi.

    - Menggunakan feature importance untuk menganalisis faktor mana yang paling berpengaruh terhadap besarnya biaya asuransi.

## Data Understanding
### 1. **Sumber Data**
Dataset yang digunakan dalam proyek ini adalah Prediction of Insurance Charges using Age, Gender, BMI, Children, Smoker, Region, yang diperoleh dari Kaggle. Dataset ini berisi 1.338 baris dan 7 fitur utama yang merepresentasikan berbagai faktor demografis dan kebiasaan hidup seseorang yang berpotensi memengaruhi besarnya biaya asuransi kesehatan yang harus dibayar.

Kaggle : https://www.kaggle.com/datasets/thedevastator/prediction-of-insurance-charges-using-age-gender

### 2. **Struktur Data**
Dataset ini berisi data pribadi peserta asuransi beserta dengan biaya tagihan asuransinya. Berdasarkan pengamatan dataset tersebut memiliki 1338 baris dan 7 kolom data.

### Variabel-variabel pada prediksi biaya asuransi dataset adalah sebagai berikut:
- Age : merupakan usia individu yang memegang polis asuransi. Variabel ini bertipe numerik dan berpengaruh langsung terhadap besarnya premi yang harus dibayar. Umumnya, semakin tua usia, semakin tinggi risikonya.
- Sex : merupakan jenis kelamin individu (male/female). Variabel ini bertipe kategorikal dan dapat memengaruhi prediksi premi karena adanya perbedaan risiko kesehatan antara pria dan wanita.
- bmi (Body Mass Index) : merupakan indeks massa tubuh seseorang, dihitung berdasarkan berat badan dan tinggi badan. Variabel ini bertipe numerik. Nilai BMI yang tinggi bisa mengindikasikan obesitas, yang berpotensi meningkatkan premi asuransi.
- Children: menunjukkan jumlah anak yang menjadi tanggungan dari pemegang polis. Bertipe numerik. Semakin banyak tanggungan, potensi biaya perawatan keluarga juga meningkat.
- Smoker: menunjukkan apakah individu merupakan perokok atau bukan (yes/no). Variabel ini bertipe kategorikal. Perokok umumnya dikenakan premi lebih tinggi karena risiko kesehatannya lebih besar.
- region : merupakan lokasi tempat tinggal pemegang polis di Amerika Serikat, dengan empat kategori: southeast, southwest, northwest, northeast. Bertipe kategorikal. Variabel ini bisa merepresentasikan perbedaan biaya layanan kesehatan antar wilayah.
- charges : merupakan target variabel dalam proyek ini, yaitu total biaya asuransi kesehatan yang harus dibayarkan individu. Bertipe numerik dan akan diprediksi menggunakan algoritma regresi.

## Data Preparation
1. Menghapus duplikasi data.
2. Mengkonversi dataset yang memiliki fitur kategorikal menjadi bentuk numerik agar bisa diproses oleh model:
    - sex diubah menjadi angka biner: 0 untuk male dan 1 untuk female
    - smoker diubah menjadi angka biner: 0 untuk no dan 1 untuk yes
    - region diubah menjadi angka (0, 1, 2, 3) menggunakan label encoding.
4. Melakukan feature scaling pada fitur numerik dengan StandardScaler agar semua fitur memiliki distribusi yang sebanding.
5. Memisahkan fitur input (X) dan target output (y) sebelum training, supaya dapat mengukur performa model secara adil.
6. Membagi data menjadi data latih dan data uji untuk mengevaluasi kinerja model.

## Modeling
Proyek ini, kita ingin memprediksi biaya asuransi (charges) berdasarkan informasi demografis dan gaya hidup seseorang. Karena variabel target bersifat numerik kontinu, maka permasalahan ini adalah regresi.

### Pemilihan Model

1. Linear Regression adalah model regresi linier yang berusaha menemukan hubungan linier antara satu atau lebih variabel input (fitur) dengan variabel output (target). Parameter yang digunakan model ini antara lain:
- fit_intercept : Jika nilai true maka model menghitung intercept (bias). Jika false maka model tidak menambahkan intercept.
- normalize : digunakan untuk menormalisasi data sebelum regresi.
- n_jobs : Jumlah core CPU yang digunakan. Jika -1, gunakan semua core. Hanya berguna di model besar.

2. Random Forest merupakan algoritma ensemble learning yang membangun banyak decision tree, lalu menggabungkan prediksi dari seluruh pohon (dalam regresi: mengambil rata-rata hasil). Algoritma ini cenderung lebih akurat daripada linear regression, dan lebih tahan terhadap overfitting karena menggunakan teknik bagging. Parameter yang digunakan model ini antara lain:
- n_estimators (nilai 100) : Jumlah pohon keputusan (decision trees) yang dibangun. Semakin banyak, semakin akurat, tapi lebih lambat.
- max_depth (nilai 10) : Maksimum kedalaman setiap pohon. Mengontrol overfitting. Nilai lebih besar = model lebih kompleks.
- random_state (nilai 42) : Supaya hasil yang diproduksi secara konsisten setiap dijalankan.

3. Gradient Boosting adalah teknik boosting di mana model dibangun secara bertahap. Setiap model baru memperbaiki kesalahan dari model sebelumnya dengan meminimalkan fungsi loss melalui gradien. Cocok untuk data kompleks dengan hubungan non-linear. Parameter yang digunakan model ini antara lain:
- n_estimators : Jika jumlah boosting stages kecil maka jumlah akan semakin kecil yang digunakan. Sebaliknya semakin besar jumlah boosting stage yang digunakan maka akan semakin kompleks.
- learning_rate : Nilai besar bisa membuat proses lebih cepat, tapi lebih berisiko. Sedangkan Nilai kecil membuat model belajar lebih lambat tapi lebih stabil dan akurat.
- max_depth : Nilai besar akan lebih akurat di data training tapi berisiko overfitting. Sedangkan nilai kecil lebih sederhana dan umum, tapi mungkin melewatkan pola penting.
- random_state (nilai 42): Supaya hasil yang diproduksi secara konsisten setiap dijalankan.


**Rubrik/Kriteria Tambahan (Opsional)**: 
Kelebihan dan Kekurangan Setiap Algoritma

A. Linear Regression
- Kelebihan:
  - Sederhana, cepat, dan mudah untuk diinterpretasi.
  - Cocok untuk hubungan linier antar fitur dan target.
- Kekurangan:
  - Tidak mampu menangkap hubungan non-linear dengan baik.
  - Sensitif terhadap outlier dan multikolinearitas antar fitur.

B. Random Forest
- Kelebihan
  - Mampu menangani hubungan non-linear dan interaksi antar fitur.
  - Tidak mudah overfitting berkat metode bagging.
  - Menyediakan feature importance untuk interpretasi fitur.
- Kekurangan
  - Lebih kompleks dan membutuhkan waktu komputasi yang lebih lama dibandingkan linear regression.
  - Kurang transparan dibandingkan model linier.

C. Gradient Boosting
- Kelebihan
  - Mampu memberikan performa tinggi dengan menggabungkan banyak model lemah (weak learners).
  - Akurat dalam prediksi dan seringkali unggul dalam kompetisi data science.
- Kekurangan
  - Lebih rentan overfitting dibandingkan Random Forest jika tidak dilakukan tuning dengan benar.
  - Proses pelatihan bisa memakan waktu lebih lama.

## Evaluation

### Metrik Evaluasi Regresi 

A. Mean Absolute Error, MAE menghitung rata-rata selisih absolut antara nilai aktual dengan nilai prediksi. Metrik ini memberikan gambaran seberapa jauh prediksi dari nilai sesungguhnya secara rata-rata dalam satuan asli.

B. Root Mean Squared Error (RMSE),
RMSE memberikan penalti lebih besar terhadap prediksi yang meleset jauh. Ini berguna untuk mengetahui seberapa besar prediksi menyimpang dari nilai sebenarnya, khususnya ketika outlier menjadi perhatian.

C. R-squared (R² Score), menunjukkan seberapa baik model menjelaskan variansi dari target (charges). Nilai R² mendekati 1 menunjukkan model yang mampu menjelaskan sebagian besar variasi data.

**Hasil Evaluasi Model**
1. Model Linear Regression menghasilkan nilai Mean Absolute Error (MAE) sebesar 0.346. Nilai Root Mean Squared Error (RMSE) sebesar 0.479 mengindikasikan adanya kesalahan prediksi yang cukup tinggi, terutama jika terdapat outlier. Sementara itu, nilai R² (koefisien determinasi) sebesar 0.783 berarti model ini mampu menjelaskan sekitar 78.3% dari variasi dalam data target. Secara keseluruhan, model ini cukup baik namun masih kurang akurat dibandingkan dua model lainnya karena keterbatasannya dalam menangkap hubungan non-linear.
2. Model Random Forest memberikan hasil yang lebih baik dengan MAE sebesar 0.208, yang berarti rata-rata kesalahan prediksi lebih kecil dibandingkan regresi linier. Nilai RMSE sebesar 0.378 menunjukkan model ini lebih mampu mengatasi nilai error besar. Dengan R² sebesar 0.865, model ini menjelaskan 86.5% dari variasi target, menunjukkan performa yang solid. 
3. Model Gradient Boosting memberikan performa terbaik di antara ketiganya, dengan MAE paling rendah yaitu 0.202, yang berarti prediksi model ini paling dekat dengan nilai aktual secara konsisten. RMSE sebesar 0.360 menunjukkan model ini juga unggul dalam meminimalkan kesalahan besar. Dengan R² sebesar 0.878, model ini mampu menjelaskan hampir 87.8% variasi pada data target. 

**Kesimpulan:**

Berdasarkan ketiga metrik evaluasi (MAE, RMSE, dan R²), Gradient Boosting Regressor dipilih sebagai model terbaik untuk digunakan dalam memprediksi biaya asuransi karena memberikan hasil prediksi paling akurat.

### Interpretasi Model: Feature Importance
**Interpretasi:**
1. Smoker memiliki pengaruh terbesar terhadap biaya asuransi. Artinya, status merokok sangat menentukan mahal atau tidaknya biaya asuransi.
2. Age dan bmi juga berkontribusi besar, menunjukkan usia dan indeks massa tubuh berpengaruh besar dalam perhitungan biaya.
3. Fitur seperti children atau region cenderung memiliki pengaruh lebih kecil.

Dengan menggunakan feature importance dari Gradient Boosting, kita bisa menyimpulkan bahwa fitur-fitur seperti smoker, age, dan bmi memiliki pengaruh paling besar terhadap prediksi biaya asuransi. Fitur lainnya seperti children atau region memberikan kontribusi yang lebih kecil terhadap model prediktif.
