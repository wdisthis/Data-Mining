# Tugas Mandiri

## Praktikum Data Mining — Seleksi Fitur dengan Metode Filter (Korelasi Pearson)

> Referensi: Modul 3 Feature Selection, Sains Data ITERA 2026 (bagian D.i, Metode Filter dengan Korelasi Pearson).

---

## 1. Soal

Terdapat data pengeluaran karyawan sebagai berikut.

| No | Umur | Tinggi (cm) | Berat (kg) | Jam Bekerja / Hari | Pengeluaran |
|----|------|-------------|------------|--------------------|-------------|
| 1 | 25 | 165 | 55 | 8 | 150 |
| 2 | 30 | 170 | 65 | 10 | 200 |
| 3 | 35 | 175 | 70 | 9 | 250 |
| 4 | 40 | 180 | 80 | 7 | 300 |
| 5 | 28 | 160 | 50 | 6 | 180 |
| 6 | 33 | 172 | 68 | 11 | 220 |
| 7 | 45 | 178 | 85 | 8 | 320 |
| 8 | 38 | 168 | 60 | 9 | 270 |
| 9 | 27 | 162 | 52 | 5 | 160 |
| 10 | 32 | 174 | 72 | 10 | 230 |

Beberapa fitur merupakan data mentah hasil survei yang tidak semuanya relevan terhadap jumlah pengeluaran. Lakukan seleksi fitur pada data di atas menggunakan **metode filter dengan Korelasi Pearson**. Fitur dianggap relevan jika memiliki nilai korelasi **lebih dari 0.5**.

| Bobot | Ketentuan |
|-------|-----------|
| [30] | Kode program dan hasil output |
| [50] | Penjelasan langkah-langkah proses seleksi |
| [20] | Fitur-fitur yang relevan terhadap target |

---

## 2. Kode Program dan Hasil Output *(bobot 30)*

### 2.1 Import library

```python
# Import library
import pandas as pd
import numpy as np
```

### 2.2 Masukkan dataset

```python
# Masukkan dataset yang akan digunakan
data = {
    'Umur': [25, 30, 35, 40, 28, 33, 45, 38, 27, 32],
    'Tinggi': [165, 170, 175, 180, 160, 172, 178, 168, 162, 174],
    'Berat': [55, 65, 70, 80, 50, 68, 85, 60, 52, 72],
    'Jam_Bekerja_Hari': [8, 10, 9, 7, 6, 11, 8, 9, 5, 10],
    'Pengeluaran': [150, 200, 250, 300, 180, 220, 320, 270, 160, 230]  # Target
}
df = pd.DataFrame(data)
print(df)
```

**Output:**

```
   Umur  Tinggi  Berat  Jam_Bekerja_Hari  Pengeluaran
0    25     165     55                 8          150
1    30     170     65                10          200
2    35     175     70                 9          250
3    40     180     80                 7          300
4    28     160     50                 6          180
5    33     172     68                11          220
6    45     178     85                 8          320
7    38     168     60                 9          270
8    27     162     52                 5          160
9    32     174     72                10          230
```

### 2.3 Hitung korelasi Pearson tiap fitur terhadap target

```python
# Hitung korelasi Pearson antar fitur dan target
corr = df.corr(method='pearson')['Pengeluaran'].drop('Pengeluaran')
```

### 2.4 Tetapkan ambang batas dan pilih fitur

```python
# Tentukan ambang batas |r| > 0.5
threshold = 0.5
selected_features = corr[abs(corr) > threshold].index.tolist()

print("Nilai Korelasi Fitur terhadap Pengeluaran:")
print(corr)
print("\nFitur yang Dipilih (|r| > 0.5):", selected_features)
```

**Output:**

```
Nilai Korelasi Fitur terhadap Pengeluaran:
Umur                0.987901
Tinggi              0.836630
Berat               0.861256
Jam_Bekerja_Hari    0.220996
Name: Pengeluaran, dtype: float64

Fitur yang Dipilih (|r| > 0.5): ['Umur', 'Tinggi', 'Berat']
```

### 2.5 Lihat hasil seleksi fitur

```python
# Dataset baru hanya berisi fitur terpilih
X_selected = df[selected_features]
y = df['Pengeluaran']

print("\nDataset setelah seleksi fitur:")
print(X_selected)
```

**Output:**

```
Dataset setelah seleksi fitur:
   Umur  Tinggi  Berat
0    25     165     55
1    30     170     65
2    35     175     70
3    40     180     80
4    28     160     50
5    33     172     68
6    45     178     85
7    38     168     60
8    27     162     52
9    32     174     72
```

---

## 3. Penjelasan Langkah-langkah Proses Seleksi *(bobot 50)*

### Langkah 1 — Memahami konsep metode filter

Metode filter memilih fitur berdasarkan ukuran statistik antara fitur dan target, **tanpa melatih model machine learning**. Pada tugas ini ukuran yang dipakai adalah **koefisien korelasi Pearson (r)**, yang cocok karena seluruh fitur dan target bertipe numerik. Nilai r berkisar dari −1 sampai +1:

- r mendekati **+1**: hubungan linear positif yang kuat (fitur naik, target ikut naik).
- r mendekati **−1**: hubungan linear negatif yang kuat (fitur naik, target turun).
- r mendekati **0**: hampir tidak ada hubungan linear.

Karena arah hubungan tidak penting untuk menilai relevansi, yang dibandingkan dengan ambang batas adalah **nilai mutlak** korelasi, yaitu |r|.

### Langkah 2 — Import library

`pandas` digunakan untuk menyimpan data dalam bentuk tabel (DataFrame) dan menghitung korelasi, sedangkan `numpy` mendukung operasi numerik. Keduanya sama dengan yang dipakai pada contoh BMI di modul.

### Langkah 3 — Memasukkan dataset dan menentukan target

Data survei dimasukkan ke dalam dictionary lalu diubah menjadi DataFrame. Pada soal ini:

- **Fitur:** `Umur`, `Tinggi`, `Berat`, dan `Jam_Bekerja_Hari`.
- **Target:** `Pengeluaran`, karena tujuan seleksi adalah mencari fitur yang relevan terhadap jumlah pengeluaran.

Kolom `No` tidak dimasukkan karena hanya nomor urut dan tidak membawa informasi apa pun tentang pengeluaran.

### Langkah 4 — Menghitung korelasi Pearson fitur terhadap target

Perintah `df.corr(method='pearson')` menghasilkan matriks korelasi antar semua kolom. Kolom `['Pengeluaran']` diambil agar hanya korelasi terhadap target yang terlihat, lalu `.drop('Pengeluaran')` membuang korelasi target dengan dirinya sendiri (yang selalu bernilai 1). Hasilnya adalah satu nilai r untuk tiap fitur.

### Langkah 5 — Menetapkan ambang batas

Soal menetapkan fitur relevan jika korelasinya **lebih dari 0.5**, sehingga `threshold = 0.5`. Perlu diperhatikan bahwa pada contoh modul tertulis `threshold = 0.8` padahal komentarnya menyebut 0.5; pada tugas ini nilai yang dipakai disesuaikan dengan soal, yaitu 0.5. Fitur dipilih dengan kondisi `abs(corr) > threshold` (menggunakan nilai mutlak, selaras dengan Langkah 1).

### Langkah 6 — Membandingkan korelasi dengan ambang batas

| Fitur | Korelasi (r) | \|r\| > 0.5 ? | Keputusan |
|-------|--------------|---------------|-----------|
| Umur | 0.987901 | Ya | **Dipilih** |
| Tinggi | 0.836630 | Ya | **Dipilih** |
| Berat | 0.861256 | Ya | **Dipilih** |
| Jam_Bekerja_Hari | 0.220996 | Tidak | Dibuang |

### Langkah 7 — Membentuk dataset hasil seleksi

Kolom yang lolos (`Umur`, `Tinggi`, `Berat`) disimpan sebagai `X_selected`, sedangkan target disimpan sebagai `y`. Dataset inilah yang nantinya dipakai untuk pelatihan model, dengan jumlah fitur berkurang dari 4 menjadi 3.

---

## 4. Fitur yang Relevan terhadap Target *(bobot 20)*

Fitur yang relevan terhadap **Pengeluaran** (|r| > 0.5) adalah:

1. **Umur** (r = 0.9879): korelasi sangat kuat. Semakin tinggi umur karyawan, semakin besar pengeluarannya.
2. **Berat** (r = 0.8613): korelasi kuat dan positif.
3. **Tinggi** (r = 0.8366): korelasi kuat dan positif.

Fitur yang **tidak relevan**:

- **Jam_Bekerja_Hari** (r = 0.2210): korelasi lemah dan berada di bawah ambang 0.5, sehingga dibuang.

### Catatan analisis

- Korelasi menunjukkan hubungan statistik, **bukan sebab-akibat**. Tinggi dan berat badan secara logika tidak menyebabkan pengeluaran naik; korelasi tinggi di sini kemungkinan muncul karena keduanya ikut naik bersama umur pada sampel yang hanya berisi 10 karyawan.
- Tinggi, Berat, dan Umur saling berkorelasi (multikolinearitas), sehingga informasi yang dibawanya bisa tumpang tindih. Metode filter korelasi tidak memeriksa hal ini; untuk itu bisa dipakai metode lanjutan seperti wrapper (Forward Selection / Backward Elimination) atau embedded.
- Dengan sampel sebanyak 10 data, hasil ini bersifat indikatif dan perlu diuji kembali pada data yang lebih besar.
