# Tugas Mandiri Modul 4 – Proximity Measurement

**Praktikum SD2531001 – Data Mining**
Program Studi Sains Data, Fakultas Sains, Institut Teknologi Sumatera, 2026

---

## A. Deskripsi Permasalahan

Perusahaan memiliki data pengeluaran 10 karyawan lama. Seorang karyawan baru bergabung dengan data: **Umur 31, Tinggi 169 cm, Berat 63 kg, Jam Kerja 9 jam/hari, dan Pengeluaran 210 ribu**. Perusahaan ingin mengetahui karyawan lama mana yang paling mirip dengan karyawan baru tersebut.

Pengukuran kemiripan dilakukan menggunakan **Euclidean Distance** dengan kelima fitur (Umur, Tinggi, Berat, Jam Kerja, Pengeluaran). Karena Euclidean Distance termasuk *dissimilarity measurement*, **semakin kecil nilai jarak, semakin mirip** kedua objek.

## B. Konsep Dasar

Euclidean distance adalah ukuran jarak lurus (garis terpendek) antara dua titik dalam ruang n-dimensi. Jarak antara objek *i* dan *j* dengan *p* fitur dihitung dengan persamaan:

```
d(i, j) = sqrt( Σ (x_ik − x_jk)² ),  k = 1 … p
```

Pada tugas ini, *i* adalah karyawan baru dan *j* adalah masing-masing karyawan lama (K1 – K10), dengan *p* = 5 fitur.

## C. Dataset

Data karyawan lama:

| No | Umur | Tinggi (cm) | Berat (kg) | Jam Kerja/Hari | Pengeluaran |
|----|------|-------------|------------|----------------|-------------|
| 1  | 25 | 165 | 55 | 8  | 150 |
| 2  | 30 | 170 | 65 | 10 | 200 |
| 3  | 35 | 175 | 70 | 9  | 250 |
| 4  | 40 | 180 | 80 | 7  | 300 |
| 5  | 28 | 160 | 50 | 6  | 180 |
| 6  | 33 | 172 | 68 | 11 | 220 |
| 7  | 45 | 178 | 85 | 8  | 320 |
| 8  | 38 | 168 | 60 | 9  | 270 |
| 9  | 27 | 162 | 52 | 5  | 160 |
| 10 | 32 | 174 | 72 | 10 | 230 |

Data karyawan baru: `[31, 169, 63, 9, 210]`

## D. Implementasi Metode

Praktikum menggunakan Google Colab dengan pustaka `pandas` dan `numpy`.

### 1) Import pustaka

```python
# Import library
import pandas as pd
import numpy as np
```

**Penjelasan:** `pandas` dipakai untuk menyimpan data dalam bentuk tabel (DataFrame), sedangkan `numpy` dipakai untuk operasi matematika seperti kuadrat dan akar. Tahap ini tidak menghasilkan output.

### 2) Masukkan dataset karyawan lama

```python
df = pd.DataFrame({
    "Umur":        [25, 30, 35, 40, 28, 33, 45, 38, 27, 32],               # Umur (tahun)
    "Tinggi":      [165, 170, 175, 180, 160, 172, 178, 168, 162, 174],     # Tinggi badan (cm)
    "Berat":       [55, 65, 70, 80, 50, 68, 85, 60, 52, 72],               # Berat badan (kg)
    "JamKerja":    [8, 10, 9, 7, 6, 11, 8, 9, 5, 10],                      # Jam bekerja per hari
    "Pengeluaran": [150, 200, 250, 300, 180, 220, 320, 270, 160, 230]      # Pengeluaran (ribu)
}, index=["K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8", "K9", "K10"])
print(df)
```

Output:

```
     Umur  Tinggi  Berat  JamKerja  Pengeluaran
K1     25     165     55         8          150
K2     30     170     65        10          200
K3     35     175     70         9          250
K4     40     180     80         7          300
K5     28     160     50         6          180
K6     33     172     68        11          220
K7     45     178     85         8          320
K8     38     168     60         9          270
K9     27     162     52         5          160
K10    32     174     72        10          230
```

**Penjelasan:** Kelima fitur dimasukkan ke dalam DataFrame `df` dengan indeks K1 sampai K10 supaya setiap baris (karyawan) mudah dikenali. Output menampilkan 10 karyawan lama dengan 5 fitur, sama persis dengan tabel pada soal.

### 3) Masukkan data karyawan baru

```python
# Karyawan baru yang akan dibandingkan dengan seluruh karyawan lama
baru = pd.Series({
    "Umur": 31,
    "Tinggi": 169,
    "Berat": 63,
    "JamKerja": 9,
    "Pengeluaran": 210
}, dtype=float)
print(baru)
```

Output:

```
Umur            31.0
Tinggi         169.0
Berat           63.0
JamKerja         9.0
Pengeluaran    210.0
dtype: float64
```

**Penjelasan:** Data karyawan baru disimpan sebagai `Series` dengan nama fitur yang sama seperti kolom `df`. Dengan nama yang sama, pandas otomatis mencocokkan fitur yang bersesuaian saat operasi pengurangan di langkah berikutnya. Tipe data `float` dipakai agar konsisten dengan perhitungan numerik pada modul.

### 4) Hitung selisih setiap karyawan lama terhadap karyawan baru

```python
# Selisih = data karyawan lama - data karyawan baru (untuk setiap fitur)
selisih = df.astype(float) - baru
print(selisih)
```

Output:

```
     Umur  Tinggi  Berat  JamKerja  Pengeluaran
K1   -6.0    -4.0   -8.0      -1.0        -60.0
K2   -1.0     1.0    2.0       1.0        -10.0
K3    4.0     6.0    7.0       0.0         40.0
K4    9.0    11.0   17.0      -2.0         90.0
K5   -3.0    -9.0  -13.0      -3.0        -30.0
K6    2.0     3.0    5.0       2.0         10.0
K7   14.0     9.0   22.0      -1.0        110.0
K8    7.0    -1.0   -3.0       0.0         60.0
K9   -4.0    -7.0  -11.0      -4.0        -50.0
K10   1.0     5.0    9.0       1.0         20.0
```

**Penjelasan:** Setiap nilai karyawan lama dikurangi nilai karyawan baru pada fitur yang sama, sesuai bagian `(x_ik − x_jk)` pada rumus. Contohnya K1: umur 25 − 31 = −6, tinggi 165 − 169 = −4, dan seterusnya. Tanda negatif berarti karyawan lama lebih kecil dari karyawan baru pada fitur tersebut, dan tanda ini tidak berpengaruh karena akan dikuadratkan.

### 5) Kuadratkan selisih

```python
# Kuadrat dari setiap selisih
kuadrat = selisih ** 2
print(kuadrat)
```

Output:

```
      Umur  Tinggi  Berat  JamKerja  Pengeluaran
K1    36.0    16.0   64.0       1.0       3600.0
K2     1.0     1.0    4.0       1.0        100.0
K3    16.0    36.0   49.0       0.0       1600.0
K4    81.0   121.0  289.0       4.0       8100.0
K5     9.0    81.0  169.0       9.0        900.0
K6     4.0     9.0   25.0       4.0        100.0
K7   196.0    81.0  484.0       1.0      12100.0
K8    49.0     1.0    9.0       0.0       3600.0
K9    16.0    49.0  121.0      16.0       2500.0
K10    1.0    25.0   81.0       1.0        400.0
```

**Penjelasan:** Selisih dipangkatkan dua agar semua nilai menjadi positif dan selisih yang besar mendapat bobot lebih besar. Terlihat kolom `Pengeluaran` memiliki nilai kuadrat jauh lebih besar dibandingkan kolom lain (misalnya K7 mencapai 12100), sehingga fitur ini akan sangat berpengaruh pada jarak akhir.

### 6) Jumlahkan kuadrat selisih pada setiap karyawan

```python
# Jumlah kuadrat selisih dari kelima fitur
jumlah_kuadrat = kuadrat.sum(axis=1)
print(jumlah_kuadrat)
```

Output:

```
K1      3717.0
K2       107.0
K3      1701.0
K4      8595.0
K5      1168.0
K6       142.0
K7     12862.0
K8      3659.0
K9      2702.0
K10      508.0
dtype: float64
```

**Penjelasan:** Kuadrat dari kelima fitur dijumlahkan per baris (`axis=1`), sesuai simbol sigma pada rumus. Contohnya K2: 1 + 1 + 4 + 1 + 100 = 107. Pada tahap ini sudah terlihat K2 (107) dan K6 (142) memiliki jumlah paling kecil, sedangkan K7 (12862) paling besar.

### 7) Hitung jarak Euclidean

```python
# Akar dari jumlah kuadrat selisih
euclidean = np.sqrt(jumlah_kuadrat)

hasil = pd.DataFrame({"Jarak_Euclidean": euclidean.round(4)})
print(hasil)
```

Output:

```
     Jarak_Euclidean
K1           60.9672
K2           10.3441
K3           41.2432
K4           92.7092
K5           34.1760
K6           11.9164
K7          113.4108
K8           60.4897
K9           51.9808
K10          22.5389
```

**Penjelasan:** Akar dari jumlah kuadrat menghasilkan jarak Euclidean, yaitu langkah terakhir dari rumus. Nilai dibulatkan 4 angka di belakang koma seperti pada modul. Contohnya K2: akar dari 107 = 10.3441. Semakin kecil jaraknya, semakin mirip karyawan tersebut dengan karyawan baru.

### 8) Urutkan dari jarak terkecil

```python
# Urutkan dari jarak terkecil (paling mirip) ke terbesar (paling berbeda)
hasil_urut = hasil.sort_values("Jarak_Euclidean")
hasil_urut["Peringkat"] = range(1, len(hasil_urut) + 1)
print(hasil_urut)
```

Output:

```
     Jarak_Euclidean  Peringkat
K2           10.3441          1
K6           11.9164          2
K10          22.5389          3
K5           34.1760          4
K3           41.2432          5
K9           51.9808          6
K8           60.4897          7
K1           60.9672          8
K4           92.7092          9
K7          113.4108         10
```

**Penjelasan:** Jarak diurutkan dari yang terkecil karena jarak kecil berarti semakin mirip, lalu setiap karyawan diberi peringkat 1 sampai 10. Hasilnya, urutan dari paling mirip ke paling berbeda adalah K2, K6, K10, K5, K3, K9, K8, K1, K4, K7.

### 9) Tentukan 3 karyawan yang paling mirip

```python
# Ambil 3 karyawan dengan jarak terkecil
top3 = hasil_urut.head(3)
print(top3)
print()
print(df.loc[top3.index])
```

Output:

```
     Jarak_Euclidean  Peringkat
K2           10.3441          1
K6           11.9164          2
K10          22.5389          3

     Umur  Tinggi  Berat  JamKerja  Pengeluaran
K2     30     170     65        10          200
K6     33     172     68        11          220
K10    32     174     72        10          230
```

**Penjelasan:** `head(3)` mengambil tiga baris teratas dari hasil yang sudah terurut, sehingga didapat K2, K6, dan K10 sebagai tiga karyawan paling mirip. Tabel kedua menampilkan data asli ketiganya untuk memperlihatkan bahwa nilai-nilainya memang dekat dengan karyawan baru (umur 30–33, tinggi 170–174, pengeluaran 200–230).

### 10) Verifikasi perhitungan manual (K2 vs karyawan baru)

```python
# Dua objek yang dibandingkan: K2 dan karyawan baru
a = df.loc["K2"].values.astype(float)     # [30, 170, 65, 10, 200]
b = baru.values                           # [31, 169, 63, 9, 210]
print(a)
print(b)

selisih_k2 = a - b                        # [-1, 1, 2, 1, -10]
kuadrat_k2 = selisih_k2 ** 2              # [1, 1, 4, 1, 100]
euclidean_k2 = np.sqrt(np.sum(kuadrat_k2))   # akar dari 107

print("Selisih  :", selisih_k2)
print("Kuadrat  :", kuadrat_k2)
print("Jumlah   :", np.sum(kuadrat_k2))
print("Euclidean:", round(euclidean_k2, 4))
```

Output:

```
[ 30. 170.  65.  10. 200.]
[ 31. 169.  63.   9. 210.]
Selisih  : [-1.  1.  2.  1. -10.]
Kuadrat  : [  1.   1.   4.   1. 100.]
Jumlah   : 107.0
Euclidean: 10.3441
```

**Penjelasan:** Perhitungan satu pasang objek dengan gaya seperti pada modul (vektor `a` dan `b`) dipakai untuk memeriksa hasil K2. Nilai akhirnya 10.3441, sama dengan hasil pada langkah 7, sehingga perhitungan untuk seluruh karyawan dinyatakan benar.

---

## E. Analisis Hasil

### Karyawan yang paling mirip

| Peringkat | Karyawan | Jarak Euclidean | Data (Umur, Tinggi, Berat, Jam, Pengeluaran) |
|-----------|----------|-----------------|----------------------------------------------|
| 1 | **K2**  | **10.3441** | 30, 170, 65, 10, 200 |
| 2 | K6  | 11.9164 | 33, 172, 68, 11, 220 |
| 3 | K10 | 22.5389 | 32, 174, 72, 10, 230 |

**Karyawan 2 (K2) memiliki kemiripan paling besar** dengan karyawan baru karena jaraknya paling kecil, yaitu 10.3441. Selisihnya sangat tipis di semua fitur: umur beda 1 tahun, tinggi beda 1 cm, berat beda 2 kg, jam kerja beda 1 jam, dan pengeluaran beda 10 ribu.

K6 berada di peringkat kedua dengan jarak 11.9164, selisihnya hanya sedikit lebih besar dari K2. Pada K6, selisih fitur fisik (berat 5 kg dan tinggi 3 cm) lebih besar daripada K2, sementara selisih pengeluarannya hanya 10 ribu. K10 berada di peringkat ketiga dengan jarak 22.5389, terutama karena selisih pengeluaran 20 ribu dan berat 9 kg.

Di sisi lain, karyawan yang paling berbeda adalah **K7** (jarak 113.4108) dan **K4** (jarak 92.7092). Keduanya memiliki pengeluaran tinggi (320 dan 300 ribu) serta umur dan berat badan yang jauh di atas karyawan baru.

### Catatan: dominasi fitur Pengeluaran

Euclidean Distance sensitif terhadap skala data. Fitur Pengeluaran memiliki rentang nilai paling besar (150–320) dibandingkan fitur lain (misalnya Jam Kerja yang hanya 5–11). Akibatnya, selisih Pengeluaran mendominasi jarak. Kontribusi kuadrat selisih Pengeluaran terhadap total jarak pada tiga karyawan terdekat:

| Karyawan | Kontribusi Pengeluaran |
|----------|------------------------|
| K2  | 93.46% |
| K10 | 78.74% |
| K6  | 70.42% |

Artinya, hasil kemiripan pada tugas ini sebagian besar ditentukan oleh seberapa dekat pengeluaran karyawan, bukan oleh kemiripan umur, postur, atau jam kerja. Jika perusahaan menginginkan kemiripan yang lebih seimbang di semua fitur, data sebaiknya dinormalisasi (misalnya Min-Max atau Z-score) sebelum menghitung jarak. Namun, sesuai ketentuan soal, perhitungan pada tugas ini menggunakan data asli tanpa normalisasi.

## F. Kesimpulan

1. Jarak Euclidean antara karyawan baru dan 10 karyawan lama berhasil dihitung menggunakan kelima fitur.
2. Urutan karyawan dari yang paling mirip: K2, K6, K10, K5, K3, K9, K8, K1, K4, K7.
3. **Tiga karyawan paling mirip adalah K2 (10.3441), K6 (11.9164), dan K10 (22.5389).**
4. **K2 adalah karyawan yang paling mirip** dengan karyawan baru.
5. Karena fitur Pengeluaran berskala paling besar, hasil ini sangat dipengaruhi oleh selisih pengeluaran. Normalisasi data dapat dipertimbangkan untuk analisis lanjutan.
