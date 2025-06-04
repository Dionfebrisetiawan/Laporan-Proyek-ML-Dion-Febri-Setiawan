# Laporan-Proyek-ML-Dion-Febri-Setiawan

## Domain Proyek: Prediksi Biaya Asuransi

Asuransi kesehatan merupakan salah satu aspek penting dalam sistem perlindungan finansial masyarakat terhadap risiko kesehatan. Perusahaan asuransi menetapkan biaya premi berdasarkan sejumlah faktor yang merepresentasikan risiko kesehatan calon pemegang polis, seperti usia, status merokok, jenis kelamin, dan indeks massa tubuh (BMI). Penentuan biaya yang tidak akurat dapat menyebabkan ketidakseimbangan antara risiko dan premi, baik merugikan perusahaan maupun pelanggan.

Dengan perkembangan teknologi dan ketersediaan data historis, machine learning dapat dimanfaatkan untuk memprediksi biaya asuransi secara lebih akurat dan efisien. Model prediktif ini dapat membantu perusahaan asuransi membuat keputusan berbasis data dalam menetapkan premi yang adil dan tepat sasaran.

Proyek ini berfokus pada pembangunan model prediksi biaya asuransi individu berdasarkan data demografis dan gaya hidup, seperti usia, jenis kelamin, status merokok, BMI, dan jumlah tanggungan. Model ini diharapkan dapat meningkatkan akurasi estimasi biaya dan mendukung proses underwriting secara otomatis.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Jelaskan mengapa dan bagaimana masalah tersebut harus diselesaikan
- Menyertakan hasil riset terkait atau referensi. Referensi yang diberikan harus berasal dari sumber yang kredibel dan author yang jelas.
- Format Referensi dapat mengacu pada penulisan sitasi [IEEE](https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/IEEE_Reference_Guide.pdf), [APA](https://www.mendeley.com/guides/apa-citation-guide/) atau secara umum seperti [di sini](https://penerbitdeepublish.com/menulis-buku-membuat-sitasi-dengan-mudah/)
- Sumber yang bisa digunakan [Scholar](https://scholar.google.com/)

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
    - Mengimplementasikan minimal dua algoritma regresi (misalnya, Linear Regression dan Random Forest Regressor) dan membandingkan performanya menggunakan metrik evaluasi seperti RMSE, MAE, dan R² Score.

    - Melakukan hyperparameter tuning pada model terbaik untuk meningkatkan performa prediksi.

    - Menggunakan feature importance atau koefisien regresi untuk menganalisis faktor mana yang paling berpengaruh terhadap besarnya biaya asuransi.

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah Prediction of Insurance Charges using Age, Gender, BMI, Children, Smoker, Region, yang diperoleh dari Kaggle. Dataset ini berisi 1.338 baris dan 7 fitur utama yang merepresentasikan berbagai faktor demografis dan kebiasaan hidup seseorang yang berpotensi memengaruhi besarnya biaya asuransi kesehatan yang harus dibayar.

Kaggle : https://www.kaggle.com/datasets/thedevastator/prediction-of-insurance-charges-using-age-gender  

### Variabel-variabel pada prediksi biaya asuransi dataset adalah sebagai berikut:
- Age : merupakan usia individu yang memegang polis asuransi. Variabel ini bertipe numerik dan berpengaruh langsung terhadap besarnya premi yang harus dibayar. Umumnya, semakin tua usia, semakin tinggi risikonya.
- Sex : merupakan jenis kelamin individu (male/female). Variabel ini bertipe kategorikal dan dapat memengaruhi prediksi premi karena adanya perbedaan risiko kesehatan antara pria dan wanita.
- bmi (Body Mass Index) : merupakan indeks massa tubuh seseorang, dihitung berdasarkan berat badan dan tinggi badan. Variabel ini bertipe numerik. Nilai BMI yang tinggi bisa mengindikasikan obesitas, yang berpotensi meningkatkan premi asuransi.
- Children: menunjukkan jumlah anak yang menjadi tanggungan dari pemegang polis. Bertipe numerik. Semakin banyak tanggungan, potensi biaya perawatan keluarga juga meningkat.
- Smoker: menunjukkan apakah individu merupakan perokok atau bukan (yes/no). Variabel ini bertipe kategorikal. Perokok umumnya dikenakan premi lebih tinggi karena risiko kesehatannya lebih besar.
- region : merupakan lokasi tempat tinggal pemegang polis di Amerika Serikat, dengan empat kategori: southeast, southwest, northwest, northeast. Bertipe kategorikal. Variabel ini bisa merepresentasikan perbedaan biaya layanan kesehatan antar wilayah.
- charges : merupakan target variabel dalam proyek ini, yaitu total biaya asuransi kesehatan yang harus dibayarkan individu. Bertipe numerik dan akan diprediksi menggunakan algoritma regresi.

**Rubrik/Kriteria Tambahan (Opsional)**:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data = pd.read_csv('insurance.csv')

data.head()

data.describe()

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

### Memastikan tidak ada data yang hilang (missing/null)
data.isnull().sum()
### Mengkonversi dataset yang memiliki fitur kategorikal menjadi bentuk numerik agar bisa diproses oleh model:
data['sex'] = data['sex'].map({'male': 0, 'female': 1})
data['smoker'] = data['smoker'].map({'no': 0, 'yes': 1})
data['region'] = data['region'].astype('category').cat.codes
### Melakukan feature scaling pada fitur numerik dengan StandardScaler agar semua fitur memiliki distribusi yang sebanding:Feature Scaling
scaler = StandardScaler()
numerical_features = ['age', 'bmi', 'children', 'charges']
data[numerical_features] = scaler.fit_transform(data[numerical_features])
### Memisahkan fitur input (X) dan target output (y) sebelum training:
X = data.drop('charges', axis=1)
y = data['charges']
### Membagi data menjadi data latih dan data uji untuk mengevaluasi kinerja model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

**Rubrik/Kriteria Tambahan (Opsional)**: 

A. Encoding Variabel Kategorikal
Variabel kategorikal tidak dapat diproses langsung oleh sebagian besar algoritma machine learning. Oleh karena itu:
- sex diubah menjadi angka biner: 0 untuk male dan 1 untuk female
- smoker diubah menjadi angka biner: 0 untuk no dan 1 untuk yes
- region diubah menjadi angka (0, 1, 2, 3) menggunakan label encoding.

B. Mengecek Missing Values
Hasilnya menunjukkan bahwa tidak ada missing value, sehingga tidak diperlukan penanganan tambahan.

C. Feature Scaling (Normalisasi)
Scaling dilakukan untuk memastikan model dapat bekerja optimal dan tidak bias terhadap fitur dengan nilai besar.

D. Pemisahan Fitur atau Target dan Pemisahan Data Latih atau Uji
Untuk memudahkan proses pelatihan, fitur (X) dan target (y) dipisahkan. Kemudian agar dapat mengukur performa model secara adil, data dibagi menjadi 80% data latih dan 20% data uji. Ini penting untuk mengevaluasi seberapa baik model mengeneralisasi data baru.

## Modeling
Proyek ini, kita ingin memprediksi biaya asuransi (charges) berdasarkan informasi demografis dan gaya hidup seseorang. Karena variabel target bersifat numerik kontinu, maka permasalahan ini adalah regresi.

- Linear Regression, model dasar dan mudah diinterpretasikan. Cocok untuk baseline.
- Random Forest Regressor, model ensambel berbasis pohon yang kuat terhadap overfitting dan mampu menangkap hubungan non-linear.
- Gradient Boosting Regressor, model boosting yang menggabungkan prediksi dari beberapa model lemah. Umumnya memberikan akurasi tinggi.

### Pelatihan Model
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    print(f"{name} → RMSE: {rmse:.3f}, R²: {r2:.3f}")

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
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

A. Mean Absolute Error, MAE menghitung rata-rata selisih absolut antara nilai aktual dengan nilai prediksi. Metrik ini memberikan gambaran seberapa jauh prediksi dari nilai sesungguhnya secara rata-rata dalam satuan asli.

B. Root Mean Squared Error (RMSE),
RMSE memberikan penalti lebih besar terhadap prediksi yang meleset jauh. Ini berguna untuk mengetahui seberapa besar prediksi menyimpang dari nilai sebenarnya, khususnya ketika outlier menjadi perhatian.

C. R-squared (R² Score), menunjukkan seberapa baik model menjelaskan variansi dari target (charges). Nilai R² mendekati 1 menunjukkan model yang mampu menjelaskan sebagian besar variasi data.

**Kesimpulan:**

Model Random Forest Regressor yang digunakan dalam proyek ini menghasilkan metrik evaluasi sebagai berikut:
- MAE: 2713.46, Rata-rata prediksi charges meleset sekitar $2713, yang masih tergolong baik dalam konteks biaya medis yang bisa sangat variatif.
- RMSE: 4495.72, Nilai RMSE yang lebih tinggi dari MAE mengindikasikan adanya beberapa outlier, namun model tetap stabil.
- R² Score: 0.86, dengan nilai R² sebesar 0.86, model ini mampu menjelaskan 86% variasi dalam data biaya asuransi berdasarkan fitur yang tersedia.
- 
**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
