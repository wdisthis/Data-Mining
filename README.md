# Modul 2 Praktikum Data Mining

**Program Studi Sains Data — Fakultas Sains — Institut Teknologi Sumatera — 2026**

---

## Tujuan Praktikum

1. Mahasiswa memahami konsep *data cleaning* dan *data integration*.
2. Mampu menerapkan teknik pembersihan data (*missing value*, *outlier*, inkonsistensi, duplikat).
3. Mampu mengintegrasikan dataset dengan *merge* dan *concatenate*.
4. Mampu melakukan analisis korelasi data numerik dan kategorik.
5. Mampu melakukan data reduksi (PCA, histogram, dan sampling) dan data transformation (*min-max normalization*, *z-score normalization*, and *normalization by decimal scaling*).

---

## Data Cleaning

Data Cleaning adalah proses penting dalam data mining yang bertujuan:

1. Memastikan kualitas data.
2. Menghapus atau memperbaiki nilai yang tidak valid, hilang, atau tidak konsisten.
3. Meningkatkan akurasi model *machine learning*.

Contoh masalah pada data:

- **a. Nilai hilang (*missing values*)**
  Beberapa cara mengatasi nilai yang hilang:
  1. Abaikan (hapus data / *record* yang mengandung nilai kosong)
  2. Isi nilai kosong secara manual
  3. Imputasi nilai secara otomatis (mean, median, modus, *end-of-tail*, suka-suka, *random sample*)
- **b. Data duplikat**
- **c. Kesalahan format** (misalnya penulisan tanggal tidak konsisten)
- **d. Outlier** (nilai ekstrem)

---

## Data Integration

Data Integration (Integrasi Data) adalah proses menggabungkan data dari berbagai sumber ke dalam satu tampilan yang koheren dan konsisten. Proses ini penting dalam data mining karena:

1. Data sering tersebar di berbagai file, database, atau sistem.
2. Integrasi yang baik meningkatkan kualitas analisis data dan hasil prediksi.

Contoh kasus:

1. Menggabungkan data pelanggan dari sistem penjualan dan sistem keuangan.
2. Menggabungkan data karyawan dari HR dan sistem kehadiran.

---

## Data Reduction

Dalam data mining, reduksi data digunakan untuk meminimalkan ukuran dataset sambil mempertahankan informasi yang paling penting. Untuk kasus data yang sangat besar, data reduksi bermanfaat untuk memperkecil ukuran data namun tetap mengandung informasi di dalamnya. Ada dua jenis data reduksi, yaitu:

1. **Dimensionality Reduction**
   Teknik ini mengurangi jumlah fitur dari dataset dengan cara menghapus fitur yang tidak relevan ataupun menggabungkan beberapa fitur menjadi satu. Beberapa contoh metode dalam *dimensionality reduction*, yaitu: *Wavelet transformation*, *Principal Component Analysis* (PCA), *feature selection*, dan *attribute subset selection*.

2. **Numerosity Reduction**
   Mengurangi volume data dengan memilih bentuk alternatif yang lebih kecil dari representasi data. *Numerosity reduction* dapat menggunakan model parametrik dan non-parametrik.
   - Metode **parametrik** (misalnya Regresi) mengasumsikan data sesuai dengan beberapa model, memperkirakan parameter model, hanya menyimpan parameter, dan membuang data (kecuali kemungkinan *outlier*).
   - Metode **non-parametrik** tidak mengasumsikan model (misalnya Histogram, Pengelompokan, dan Pengambilan Sampel).

Data reduction merupakan langkah penting dalam penambangan data, karena dapat membantu meningkatkan efisiensi dan kinerja algoritma pembelajaran mesin dengan mengurangi ukuran dataset. Namun, penting untuk menyadari adanya pertimbangan antara ukuran dan akurasi data, serta menilai dengan cermat risiko dan manfaatnya sebelum menerapkannya.

---

## Data Transformation

Dalam konteks penambangan data, transformasi data adalah tindakan mengubah data mentah menjadi format yang sesuai untuk analisis dan pemodelan. Transformasi data bertujuan untuk mempersiapkan data untuk penambangan data guna mengekstrak pengetahuan dan wawasan yang berharga. Transformasi data merupakan fase penting dalam proses penambangan data karena menjamin bahwa data bebas dari kesalahan dan inkonsistensi serta dalam format yang dapat digunakan untuk analisis dan pemodelan. Dengan menurunkan dimensi data dan menskalakannya ke rentang nilai umum, transformasi data juga dapat membantu meningkatkan kinerja algoritma penambangan data.

Salah satu transformasi data dalam Data Mining adalah **Data Normalization**. Proses ini melibatkan pengubahan semua variabel data ke dalam rentang tertentu, biasanya antara [0,1]. Teknik yang digunakan untuk normalisasi adalah Normalisasi Min-Max, Normalisasi Z-Score, dan Normalisasi dengan Skala Desimal.

---

# Prosedur Praktikum Minggu 2

## ➢ Data Cleaning

### 1. Import library yang diperlukan untuk data cleaning

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

### 2. Input dataset praktikum

```python
data = {
    "ID": [1, 2, 3, 4, 5, 6],
    "Nama": ["Andi", "Budi", "Cici", "Dedi", None, "Fina"],
    "Umur": [23, 25, np.nan, 22, 200, np.nan],  # missing + outlier
    "Gaji": [4000, 4200, 4100, None, 4300, 3900]
}
df = pd.DataFrame(data)
print("Data Awal:\n", df)
```

Output:

```
Data Awal:
    ID  Nama   Umur    Gaji
0   1  Andi   23.0  4000.0
1   2  Budi   25.0  4200.0
2   3  Cici    NaN  4100.0
3   4  Dedi   22.0     NaN
4   5  None  200.0  4300.0
5   6  Fina    NaN  3900.0
```

### 3. Menampilkan jumlah nilai yang hilang di setiap kolom

```python
df.isnull().sum() #menampilkan jumlah nilai yang hilang di setiap kolom
```

Output:

```
ID       0
Nama     1
Umur     2
Gaji     1
dtype: int64
```

### 4. Menghapus baris untuk menghapus data kosong

```python
#Menghapus baris untuk menghapus data kosong
df_dropped = df.dropna()
print("\nData setelah menghapus baris dengan nilai yang hilang:\n", df_dropped)
```

Output:

```
Data setelah menghapus baris dengan nilai yang hilang:
    ID  Nama  Umur    Gaji
0   1  Andi  23.0  4000.0
1   2  Budi  25.0  4200.0
```

### 5. Menangani data yang hilang menggunakan mean

```python
# Mean
df_mean = df.copy()
df_mean["Umur"].fillna(df_mean["Umur"].mean(), inplace=True)
df_mean["Gaji"].fillna(df_mean["Gaji"].mean(), inplace=True)

print("\nHasil Imputasi - Mean:\n", df_mean)
```

Output:

```
Hasil Imputasi - Mean:
    ID  Nama   Umur    Gaji
0   1  Andi   23.0  4000.0
1   2  Budi   25.0  4200.0
2   3  Cici   67.5  4100.0
3   4  Dedi   22.0  4100.0
4   5  None  200.0  4300.0
5   6  Fina   67.5  3900.0
```

### 6. Menangani data yang hilang menggunakan median

```python
# Median
df_median = df.copy()
df_median["Umur"].fillna(df_median["Umur"].median(), inplace=True)
df_median["Gaji"].fillna(df_median["Gaji"].median(), inplace=True)
print("\nHasil Imputasi - Median:\n", df_median)
```

Output:

```
Hasil Imputasi - Median:
    ID  Nama   Umur    Gaji
0   1  Andi   23.0  4000.0
1   2  Budi   25.0  4200.0
2   3  Cici   24.0  4100.0
3   4  Dedi   22.0  4100.0
4   5  None  200.0  4300.0
5   6  Fina   24.0  3900.0
```

### 7. Menangani data yang hilang menggunakan modus

```python
# Modus
df_mode = df.copy()
df_mode["Umur"].fillna(df_mode["Umur"].mode()[0], inplace=True)
df_mode["Gaji"].fillna(df_mode["Gaji"].mode()[0], inplace=True)
print("\nHasil Imputasi - Modus:\n", df_mode)
```

Output:

```
Hasil Imputasi - Modus:
    ID  Nama   Umur    Gaji
0   1  Andi   23.0  4000.0
1   2  Budi   25.0  4200.0
2   3  Cici   22.0  4100.0
3   4  Dedi   22.0  3900.0
4   5  None  200.0  4300.0
5   6  Fina   22.0  3900.0
```

### 8. Menangani data yang hilang menggunakan konstanta atau nilai suka-suka

```python
# Konstanta
df_const = df.copy()
df_const["Umur"].fillna(30, inplace=True)
df_const["Gaji"].fillna(4000, inplace=True)
print("\nHasil Imputasi - Konstanta:\n", df_const)
```

Output:

```
Hasil Imputasi - Konstanta:
    ID  Nama   Umur    Gaji
0   1  Andi   23.0  4000.0
1   2  Budi   25.0  4200.0
2   3  Cici   30.0  4100.0
3   4  Dedi   22.0  4000.0
4   5  None  200.0  4300.0
5   6  Fina   30.0  3900.0
```

### 9. Menangani data yang hilang menggunakan *End-of-Tail*

```python
# End of Tail
df_tail = df.copy()
umur_fill = df_tail["Umur"].mean() + 3 * df_tail["Umur"].std()
df_tail["Umur"].fillna(umur_fill, inplace=True)
gaji_fill = df_tail["Gaji"].mean() - 3 * df_tail["Gaji"].std()
df_tail["Gaji"].fillna(gaji_fill, inplace=True)
print("\nHasil Imputasi - End of Tail:\n", df_tail)
```

Output:

```
Hasil Imputasi - End of Tail:
    ID  Nama        Umur         Gaji
0   1  Andi   23.000000  4000.000000
1   2  Budi   25.000000  4200.000000
2   3  Cici  332.526414  4100.000000
3   4  Dedi   22.000000  3625.658351
4   5  None  200.000000  4300.000000
5   6  Fina  332.526414  3900.000000
```

### 10. Menangani data yang hilang menggunakan Random Sample

```python
# Random Sample
df_random = df.copy()
df_random["Umur"] = df_random["Umur"].apply(
    lambda x: np.random.choice(df_random["Umur"].dropna()) if pd.isnull(x) else x
)
df_random["Gaji"] = df_random["Gaji"].apply(
    lambda x: np.random.choice(df_random["Gaji"].dropna()) if pd.isnull(x) else x
)
print("\nHasil Imputasi - Random Sample:\n", df_random)
```

Output:

```
Hasil Imputasi - Random Sample:
    ID  Nama   Umur    Gaji
0   1  Andi   23.0  4000.0
1   2  Budi   25.0  4200.0
2   3  Cici   22.0  4100.0
3   4  Dedi   22.0  4300.0
4   5  None  200.0  4300.0
5   6  Fina   25.0  3900.0
```

> Hasil *random sample* dapat berbeda pada setiap eksekusi.

### 11. Deteksi outlier menggunakan visualisasi

```python
# Gunakan data hasil imputasi mean
df_clean = df_mean.copy()

plt.figure(figsize=(6,4))
sns.boxplot(data=df_clean[["Umur", "Gaji"]])
plt.title("Boxplot untuk Deteksi Outlier")
plt.show()
```

> **Output:** *Boxplot* dengan judul "Boxplot untuk Deteksi Outlier". Kolom **Gaji** berada di sekitar 3900–4300, sedangkan kolom **Umur** berada di dekat 0 dengan satu titik *outlier* di sekitar 200.

### 12. Menangani outlier

```python
Q1 = df_clean["Umur"].quantile(0.25)
Q3 = df_clean["Umur"].quantile(0.75)
IQR = Q3 - Q1
batas_bawah = Q1 - 1.5 * IQR
batas_atas = Q3 + 1.5 * IQR

df_no_outlier = df_clean[(df_clean["Umur"] >= batas_bawah) & (df_clean["Umur"] <= batas_atas)]
print("\nData tanpa Outlier:\n", df_no_outlier)
```

Output:

```
Data tanpa Outlier:
    ID  Nama  Umur    Gaji
0   1  Andi  23.0  4000.0
1   2  Budi  25.0  4200.0
2   3  Cici  67.5  4100.0
3   4  Dedi  22.0  4100.0
5   6  Fina  67.5  3900.0
```

### 13. Menangani data yang tidak konsisten

#### a. Kasus penulisan nama kota yang berbeda

```python
data_kota = {
    "ID": [1, 2, 3, 4, 5],
    "Kota": ["Jakarta", "jakarta", "JKT", "Bandung", "bdg"]
}
df_kota = pd.DataFrame(data_kota)
df_kota["Kota"] = df_kota["Kota"].str.lower()
mapping_kota = {"jakarta": "Jakarta", "jkt": "Jakarta", "bandung": "Bandung", "bdg": "Bandung"}
df_kota["Kota"] = df_kota["Kota"].replace(mapping_kota)
print("\nData Kota Setelah Normalisasi:\n", df_kota)
```

Output:

```
Data Kota Setelah Normalisasi:
    ID     Kota
0   1  Jakarta
1   2  Jakarta
2   3  Jakarta
3   4  Bandung
4   5  Bandung
```

#### b. Kasus format tanggal yang berbeda

```python
data_tanggal = {
    "ID": [1, 2, 3],
    "Tanggal": ["2025-01-05", "05/01/2025", "Jan 5, 2025"]
}
df_tgl = pd.DataFrame(data_tanggal)
df_tgl["Tanggal"] = pd.to_datetime(df_tgl["Tanggal"], errors="coerce")
print("\nData Tanggal Setelah Normalisasi:\n", df_tgl)
```

Output:

```
Data Tanggal Setelah Normalisasi:
    ID    Tanggal
0   1 2025-01-05
1   2        NaT
2   3        NaT
```

### 14. Mengatasi data duplikat

```python
data_duplikat = {
    "ID": [1, 2, 2, 3, 4, 4],
    "Nama": ["Andi", "Budi", "Budi", "Cici", "Dedi", "Dedi"],
    "Umur": [23, 25, 25, 22, 24, 24]
}
df_dup = pd.DataFrame(data_duplikat)
print("\nJumlah Duplikat:", df_dup.duplicated().sum())
df_no_dup = df_dup.drop_duplicates(keep="first")
print("\nData Setelah Hapus Duplikat:\n", df_no_dup)
```

Output:

```
Jumlah Duplikat: 2

Data Setelah Hapus Duplikat:
    ID  Nama  Umur
0   1  Andi    23
1   2  Budi    25
3   3  Cici    22
4   4  Dedi    24
```

---

## ➢ Data Integration

### 1. Menggabungkan 2 datasets

```python
data_gaji = {
    "ID": [1, 2, 3, 4, 5, 6],
    "Bonus": [500, 600, 550, 650, 700, 580],
    "Departemen": ["IT", "HR", "Finance", "IT", "HR", "Finance"]
}
df_gaji = pd.DataFrame(data_gaji)

df_merge = pd.merge(df_no_outlier, df_gaji, on="ID", how="inner")
print("\nData Hasil Merge:\n", df_merge)
```

Output:

```
Data Hasil Merge:
    ID  Nama  Umur    Gaji  Bonus Departemen
0   1  Andi  23.0  4000.0    500         IT
1   2  Budi  25.0  4200.0    600         HR
2   3  Cici  67.5  4100.0    550    Finance
3   4  Dedi  22.0  4100.0    650         IT
4   6  Fina  67.5  3900.0    580    Finance
```

### 2. Menambahkan baris pada datasets

```python
data_cabang = {
    "ID": [7, 8],
    "Nama": ["Gilang", "Hana"],
    "Umur": [28, 27],
    "Gaji": [4500, 4700],
    "Bonus": [600, 650],
    "Departemen": ["IT", "HR"]
}
df_cabang = pd.DataFrame(data_cabang)

df_final = pd.concat([df_merge, df_cabang], ignore_index=True)
print("\nData Setelah Concatenate:\n", df_final)
```

Output:

```
Data Setelah Concatenate:
    ID    Nama  Umur    Gaji  Bonus Departemen
0   1    Andi  23.0  4000.0    500         IT
1   2    Budi  25.0  4200.0    600         HR
2   3    Cici  67.5  4100.0    550    Finance
3   4    Dedi  22.0  4100.0    650         IT
4   6    Fina  67.5  3900.0    580    Finance
5   7  Gilang  28.0  4500.0    600         IT
6   8    Hana  27.0  4700.0    650         HR
```

### 3. Analisis korelasi menggunakan Pearson Correlation dan Covariance (Data Numerik)

```python
data_num = {
    "Umur": [23, 25, 22, 30, 28, 35, 40, 41, 29, 33],
    "Gaji": [4000, 4200, 3900, 5200, 5000, 6000, 7500, 7800, 4900, 5500],
    "Pengeluaran": [2000, 2100, 1800, 3000, 2800, 3500, 4000, 4200, 2700, 3100]
}
df_num = pd.DataFrame(data_num)

print("\nPearson Correlation:\n", df_num.corr(method="pearson"))
print("\nCovariance:\n", df_num.cov())
```

Output:

```
Pearson Correlation:
                  Umur      Gaji  Pengeluaran
Umur         1.000000  0.985049     0.991163
Gaji         0.985049  1.000000     0.979480
Pengeluaran  0.991163  0.979480     1.000000

Covariance:
                    Umur          Gaji   Pengeluaran
Umur           43.822222  8.866667e+03  5.364444e+03
Gaji         8866.666667  1.848889e+06  1.088889e+06
Pengeluaran  5364.444444  1.088889e+06  6.684444e+05
```

### 4. Visualisasi Pearson menggunakan Heatmap

```python
df_num = pd.DataFrame(data_num)

# Heatmap korelasi Pearson
plt.figure(figsize=(6,4))
sns.heatmap(df_num.corr(method="pearson"), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heatmap Korelasi Numerik (Pearson)")
plt.show()
```

> **Output:** *Heatmap* korelasi Pearson 3×3 (Umur, Gaji, Pengeluaran). Diagonal bernilai 1.00; Umur–Gaji 0.99; Umur–Pengeluaran 0.99; Gaji–Pengeluaran 0.98.

### 5. Visualisasi Pairplot untuk melihat hubungan numerik

```python
sns.pairplot(df_num)
plt.suptitle("Pairplot Antar Variabel Numerik", y=1.02)
plt.show()
```

> **Output:** *Pairplot* antar variabel Umur, Gaji, dan Pengeluaran. Diagonal berisi histogram, sedangkan sisanya berupa *scatter plot* yang menunjukkan hubungan linear positif yang kuat.

### 6. Analisis korelasi menggunakan Chi-Square test (Data Nominal)

```python
import scipy.stats as stats

# Contoh data kategorik
data_cat = {
    "Gender": ["Pria", "Wanita", "Pria", "Wanita", "Pria", "Wanita", "Pria", "Wanita", "Pria", "Wanita"],
    "Pembelian": ["Ya", "Tidak", "Ya", "Ya", "Tidak", "Tidak", "Ya", "Ya", "Tidak", "Ya"]
}

df_cat = pd.DataFrame(data_cat)
print("Data Kategorik:\n", df_cat)

# --- Buat tabel kontingensi ---
contingency_table = pd.crosstab(df_cat["Gender"], df_cat["Pembelian"])
print("\nTabel Kontingensi:\n", contingency_table)

# --- Uji Chi-Square ---
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

print("\nChi-Square Test Result:")
print("Chi2 Statistic:", chi2)
print("p-value:", p)
print("Degrees of Freedom:", dof)
print("Expected Frequencies:\n", expected)

# --- Interpretasi hasil ---
if p < 0.05:
    print("\nKesimpulan: Ada hubungan signifikan antara Gender dan Pembelian.")
else:
    print("\nKesimpulan: Tidak ada hubungan signifikan antara Gender dan Pembelian.")
```

Output:

```
Data Kategorik:
   Gender Pembelian
0    Pria        Ya
1  Wanita     Tidak
2    Pria        Ya
3  Wanita        Ya
4    Pria     Tidak
5  Wanita     Tidak
6    Pria        Ya
7  Wanita        Ya
8    Pria     Tidak
9  Wanita        Ya

Tabel Kontingensi:
 Pembelian  Tidak  Ya
Gender
Pria           2   3
Wanita         2   3

Chi-Square Test Result:
Chi2 Statistic: 0.0
p-value: 1.0
Degrees of Freedom: 1
Expected Frequencies:
 [[2. 3.]
 [2. 3.]]

Kesimpulan: Tidak ada hubungan signifikan antara Gender dan Pembelian.
```

### 7. Visualisasi Mosaic Plot pada Chi-Square test

```python
from statsmodels.graphics.mosaicplot import mosaic
# Mosaic Plot
plt.figure(figsize=(6,4))
mosaic(df_cat, ["Gender", "Pembelian"])
plt.title("Mosaic Plot: Gender vs Pembelian")
plt.show()
```

> **Output:** *Mosaic plot* "Gender vs Pembelian" dengan empat kotak (Pria–Tidak, Wanita–Tidak, Pria–Ya, Wanita–Ya) berukuran proporsional.

---

## ➢ Data Reduction

### 1. PCA

PCA dapat menggunakan pustaka Sklearn di Python. Sebelum memulai, Anda harus mengimpor beberapa pustaka dan membuat dataset data iris menggunakan distribusi normal. Data iris berisi 4 fitur, seperti pada kode berikut.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#Set random seed untuk reproduksibilitas
np.random.seed(0)

#Buat dataset dengan fitur yang terstruktur menggunakan distribusi normal
data = {
    'Panjang Petal': np.random.normal(loc=3.5, scale=0.5, size=50),
    'Lebar Petal': np.random.normal(loc=1.0, scale=0.3, size=50),
    'Panjang Sepal': np.random.normal(loc=5.0, scale=0.5, size=50),
    'Lebar Sepal': np.random.normal(loc=1.5, scale=0.3, size=50)
}
df = pd.DataFrame(data)
print(df)
```

Output (cuplikan 26 baris pertama):

```
    Panjang Petal  Lebar Petal  Panjang Sepal  Lebar Sepal
0        4.382026     0.731360       5.941575     1.479528
1        3.700079     1.116071       4.326120     2.014003
2        3.989369     0.846758       4.364758     1.276574
3        4.620447     0.645810       5.484698     1.252068
4        4.433779     0.991545       4.413438     1.470464
5        3.011361     1.128500       5.971811     1.300957
6        3.975044     1.019955       4.793191     1.837991
7        3.424321     1.090742       4.626273     1.176021
8        3.448391     0.809703       5.961471     1.155759
9        3.705299     0.891178       5.740257     1.368654
10       3.572022     0.798262       5.933779     1.350590
11       4.227137     0.892134       5.453022     2.078860
12       3.880519     0.756056       4.569387     1.784826
13       3.560838     0.482115       5.955032     1.526265
14       3.721932     1.053228       4.865998     1.132369
15       3.666837     0.879466       5.401228     1.753309
16       4.247040     0.510940       5.473626     1.199935
17       3.397421     1.138835       4.922495     1.036569
18       3.656534     0.727810       5.307040     1.856409
19       3.072952     1.015584       5.461103     1.595083
20       2.223505     1.218727       5.188213     1.776258
21       3.826809     1.038695       4.450300     1.595618
22       3.932218     1.341820       5.149119     1.757049
23       3.128917     0.629552       5.663193     1.304692
24       4.634877     1.120702       4.652716     1.189727
25       2.772817     0.794557       4.925183     1.704478
```

Kemudian, visualisasikan dataset yang Anda peroleh. Harap dicatat bahwa setiap praktisi mungkin memiliki dataset yang berbeda, karena dataset dihasilkan menggunakan *random generator*.

```python
#Visualisasikan dataset
plt.figure(figsize=(12, 6))

#Scatter plot untuk data asli (Panjang Petal vs Lebar Petal)
plt.subplot(1, 2, 1)
plt.scatter(df['Panjang Petal'], df['Lebar Petal'], color='blue', marker='o')
plt.title('Data Awal (Panjang Petal vs Lebar Petal)')
plt.xlabel('Panjang Petal')
plt.ylabel('Lebar Petal')
plt.grid()
plt.tight_layout()
plt.show()
```

> **Output:** *Scatter plot* "Data Awal (Panjang Petal vs Lebar Petal)" — titik-titik biru tersebar tanpa pola linear yang jelas (sumbu x ≈ 2.2–4.7, sumbu y ≈ 0.5–1.6).

Kemudian, kita dapat melakukan dekomposisi fitur menggunakan Analisis Komponen Utama dan memvisualisasikan PCA dengan menggunakan kode berikut.

```python
#Lakukan PCA menggunakan scikit-learn
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df)

#Buat DataFrame untuk hasil PCA
df_pca = pd.DataFrame(data=X_pca, columns=['Komponen 1', 'Komponen 2'])
print(df_pca)

#Visualisasikan hasil PCA
plt.figure(figsize=(8, 6))
plt.scatter(df_pca['Komponen 1'], df_pca['Komponen 2'], color = 'green')
plt.title('Hasil PCA')
plt.xlabel('Komponen 1')
plt.ylabel('Komponen 2')
plt.grid()

plt.tight_layout()
plt.show()
```

Hasil yang diperoleh sebagai berikut (cuplikan 27 baris pertama):

```
    Komponen 1  Komponen 2
0     1.059129    0.459223
1    -0.237823   -0.858861
2     0.143784   -0.818869
3     1.145992   -0.012863
4     0.540615   -0.978556
5    -0.202119    0.990609
6     0.210453   -0.505841
7    -0.288690   -0.376070
8     0.235043    0.862233
9     0.363469    0.529974
10    0.318145    0.767525
11    0.662264   -0.016113
12    0.065761   -0.644798
13    0.316406    0.797746
14    0.079932   -0.260667
15    0.163401    0.188812
16    0.811725    0.136223
17   -0.194343   -0.082501
18    0.118299    0.107496
19   -0.354797    0.474843
20   -1.271430    0.505892
21   -0.023972   -0.735307
22    0.287115   -0.183107
23   -0.172455    0.708021
24    0.835651   -0.812990
25   -0.822898    0.104640
26   -0.127709   -0.249508
```

> **Output:** *Scatter plot* "Hasil PCA" (titik hijau) dengan sumbu Komponen 1 (≈ −1.3 s.d. 1.2) dan Komponen 2 (≈ −1.0 s.d. 1.0).

### 2. Histogram Analysis

Merepresentasikan distribusi data menggunakan rentang nilai (*bin*) dan frekuensi kemunculannya.

```python
import seaborn as sns

# Histogram untuk setiap kolom fitur
for col in df.columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, bins=20, color='skyblue')
    plt.title(f'Distribusi Atribut: {col}')
    plt.xlabel(col)
    plt.ylabel('Frekuensi')
    plt.grid(True)
    plt.show()
```

> **Output:** Histogram "Distribusi Atribut: Panjang Petal" dengan kurva KDE (frekuensi tertinggi 6 di sekitar nilai 3.4 dan 3.6).

Hasil histogram terlihat hampir terdistribusi normal.

### 3. Sampling

Simple random sampling dapat menggunakan kode berikut.

```python
# Sampling 30% data secara acak
sampled_data = df.sample(frac=0.3)
print("Jumlah data setelah sampling:", len(sampled_data))
print(sampled_data)
```

Output:

```
Jumlah data setelah sampling: 15
    Panjang Petal  Lebar Petal  Panjang Sepal  Lebar Sepal
15       3.666837     0.879466       5.401228     1.753309
25       2.772817     0.794557       4.925183     1.704478
39       3.348849     1.316336       4.453469     1.962904
16       4.247040     0.510940       5.473626     1.199935
22       3.932218     1.341820       5.149119     1.757049
29       4.234679     1.016850       5.203731     1.505244
4        4.433779     0.991545       4.413438     1.470464
1        3.700079     1.116071       4.326120     2.014003
18       3.656534     0.727810       5.307040     1.856409
12       3.880519     0.756056       4.569387     1.784826
13       3.560838     0.482115       5.955032     1.526265
17       3.397421     1.138835       4.922495     1.036569
40       2.975724     0.879047       4.254371     1.112143
26       3.522879     0.738761       4.782423     1.258977
32       3.056107     1.139699       4.662834     1.306914
```

Sampling *without replacement* dapat menggunakan kode berikut.

```python
# Sampling 30% of the data without replacement
sampled_data_no_replacement = df.sample(frac=0.3, random_state=42, replace=False)
print("Number of data points after sampling without replacement:", len(sampled_data_no_replacement))
print(sampled_data_no_replacement)
```

Output:

```
Number of data points after sampling without replacement: 15
    Panjang Petal  Lebar Petal  Panjang Sepal  Lebar Sepal
13       3.560838     0.482115       5.955032     1.526265
39       3.348849     1.316336       4.453469     1.962904
30       3.577474     0.650455       4.615042     1.393802
45       3.280963     1.211972       5.472240     1.448536
17       3.397421     1.138835       4.922495     1.036569
48       2.693051     1.038074       4.342046     2.148971
26       3.522879     0.738761       4.782423     1.258977
25       2.772817     0.794557       4.925183     1.704478
32       3.056107     1.139699       4.662834     1.306914
19       3.072952     1.015584       5.461103     1.595083
12       3.880519     0.756056       4.569387     1.784826
4        4.433779     0.991545       4.413438     1.470464
37       4.101190     0.946023       4.895851     1.515650
8        3.448391     0.809703       5.961471     1.155759
3        4.620447     0.645810       5.484698     1.252068
```

Sampling *with replacement* dapat menggunakan kode berikut.

```python
# Sampling 30% of the data with replacement
sampled_data_with_replacement = df.sample(frac=0.3, random_state=42, replace=True)
print("Number of data points after sampling with replacement:", len(sampled_data_with_replacement))
print(sampled_data_with_replacement)
```

Output:

```
Number of data points after sampling with replacement: 15
    Panjang Petal  Lebar Petal  Panjang Sepal  Lebar Sepal
38       3.306337     0.678774       5.198003     1.278131
28       4.266390     0.906534       5.336147     1.363340
14       3.721932     1.053228       4.865998     1.132369
42       2.646865     1.062482       5.083337     1.488215
7        3.424321     1.090742       4.626273     1.176021
20       2.223505     1.218727       5.188213     1.776258
38       3.306337     0.678774       5.198003     1.278131
18       3.656534     0.727810       5.307040     1.856409
22       3.932218     1.341820       5.149119     1.757049
10       3.572022     0.798262       5.933779     1.350590
10       3.572022     0.798262       5.933779     1.350590
23       3.128917     0.629552       5.663193     1.304692
35       3.578174     1.568767       5.338217     1.019383
39       3.348849     1.316336       4.453469     1.962904
23       3.128917     0.629552       5.663193     1.304692
```

Dapat diamati bahwa hasil yang diperoleh dari *random sampling*, *sampling without replacement*, dan *sampling with replacement* berbeda. Hal ini disebabkan oleh cara pengambilan sampel yang berbeda.

---

## ➢ Data Transformation

### 1. Min-Max Normalization

$$v' = \frac{v - min_A}{max_A - min_A}$$

Misalkan: $min_A$ adalah nilai minimum dan $max_A$ adalah nilai maksimum dari suatu atribut. $v$ adalah nilai yang ingin Anda plot dalam rentang baru. $v'$ adalah nilai baru yang Anda dapatkan setelah menormalisasi nilai lama. Normalisasi dapat dilakukan menggunakan rumus di atas.

```python
import pandas as pd

#Contoh DataFrame
data = {
    'A': [75, 68, 80, 90, 69],
    'B': [15, 29, 80, 26, 57]
}
df = pd.DataFrame(data)
print(df)

#Normalisasi min-max
df_normalized = (df - df.min()) / (df.max() - df.min())
print("Data Normalized (Min-Max):\n", df_normalized)
```

Output:

```
    A   B
0  75  15
1  68  29
2  80  80
3  90  26
4  69  57
Data Normalized (Min-Max):
           A         B
0  0.318182  0.000000
1  0.000000  0.215385
2  0.545455  1.000000
3  1.000000  0.169231
4  0.045455  0.646154
```

Min-Max Normalization juga dapat dilakukan menggunakan pustaka `MinMaxScaler` pada Python.

```python
from sklearn.preprocessing import MinMaxScaler

#Inisialisasi MinMaxScaler
scaler = MinMaxScaler()

#Normalisasi menggunakan Scikit-learn
df_normalized1 = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
print("Data Normalized (Min-Max) dengan Scikit-learn:\n", df_normalized1)
```

Output:

```
Data Normalized (Min-Max) dengan Scikit-learn:
           A         B
0  0.318182  0.000000
1  0.000000  0.215385
2  0.545455  1.000000
3  1.000000  0.169231
4  0.045455  0.646154
```

### 2. Z-Score Normalization

$$v' = \frac{v - mean(A)}{Standard\ Deviation\ (A)}$$

Pada Z-score Normalization, nilai-nilai suatu atribut A dinormalisasi berdasarkan rata-rata A dan deviasi standarnya. Nilai $v$ dari atribut A dinormalisasi menjadi $v'$. Normalisasi Skor Z dalam Python menggunakan rumus di atas dapat menggunakan kode berikut.

```python
#Menghitung rata-rata (mean) dan deviasi standar (standard deviation)
mean = df.mean()
std_dev = df.std()

#Menghitung Z-score menggunakan rumus
df_zscore1 = (df - mean) / std_dev

print("Data Normalized (Z-Score):\n", df_zscore1)
```

Output:

```
Data Normalized (Z-Score):
           A         B
0 -0.155268 -0.994070
1 -0.931610 -0.466912
2  0.399261  1.453451
3  1.508321 -0.579874
4 -0.820704  0.587405
```

Z-score Normalization juga dapat dilakukan menggunakan pustaka `StandardScaler` pada Python.

```python
#Inisialisasi StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#Melakukan standarisasi
df_standardized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
print("Data Normalized (Z-Score) dengan Scikit-learn:\n", df_standardized)
```

Output:

```
Data Normalized (Z-Score) dengan Scikit-learn:
           A         B
0 -0.173595 -1.111404
1 -1.041571 -0.522023
2  0.446388  1.625007
3  1.686354 -0.648319
4 -0.917575  0.656739
```

> **Catatan:** Hasil `df.std()` (pandas) memakai simpangan baku sampel (*ddof=1*), sedangkan `StandardScaler` memakai simpangan baku populasi (*ddof=0*), sehingga nilai keduanya sedikit berbeda.

### 3. Normalization by Decimal Scaling

Fungsi ini menormalkan nilai suatu atribut dengan mengubah posisi titik desimalnya. Jumlah titik desimal yang digeser dapat ditentukan oleh nilai maksimum absolut dari atribut A. Nilai $v$ dari atribut A dinormalisasi menjadi $v'$ dengan menghitung

$$v' = \frac{v}{10^j}$$

dimana $j$ adalah bilangan bulat terkecil sedemikian sehingga $Max(|v'|) < 1$.

```python
import pandas as pd

#Contoh DataFrame
data = {
    'A': [123, 456, 789, 1011, 1213],
    'B': [50, 200, 300, 400, 500]
}
df1 = pd.DataFrame(data)

#Menentukan nilai maksimum
max_value = df1.max().max()

#Menentukan j (jumlah digit dari nilai maksimum)
j = len(str(max_value))

#Melakukan normalisasi dengan decimal scaling
df1_normalized = df1 / (10 ** j)
print("Data Normalized (Decimal Scaling):\n", df1_normalized)
```

Output:

```
Data Normalized (Decimal Scaling):
        A      B
0  0.0123  0.005
1  0.0456  0.020
2  0.0789  0.030
3  0.1011  0.040
4  0.1213  0.050
```
