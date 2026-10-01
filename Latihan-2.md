# Tugas Individu 2 â€” Praktikum Data Mining

**Nama:** ____________________  **NIM:** ____________________
**Dataset:** `dataset_praktikum2.csv` (200 baris; kolom `ID`, `Nama`, `Umur`, `Gaji`, `Kota`)
**Acuan:** Modul 2 Praktikum Data Mining (hlm. 4â€“22)

---

## Catatan Penting Sebelum Memakai Jawaban Ini

- File `dataset_praktikum2.csv` **tidak ikut terunggah** ke saya (hanya PDF soalnya). Karena itu, jawaban ini berisi **kode lengkap + penjelasan konsep + kode yang mencetak interpretasi otomatis**, tetapi **angka hasil aktualnya belum bisa saya tuliskan**.
- Kode sudah saya uji pada data tiruan dengan struktur kolom yang sama (ada NaN, outlier, ID ganda, dan penulisan Kota tidak seragam), kecuali bagian *mosaic plot* karena pustaka `statsmodels` tidak tersedia di lingkungan uji saya (kodenya mengikuti modul hlm. 13).
- Bagian bertanda **âœï¸ Isi dari hasil run-mu** harus kamu lengkapi dengan angka dari notebook-mu sendiri. Kalau kamu unggah CSV-nya ke chat ini, saya bisa isikan semuanya.
- Jalankan sel **secara berurutan** dari atas ke bawah, karena Soal 2â€“5 memakai hasil Soal 1.
- Simpan notebook dengan nama `TugasIndividu2_DataMining_NIM.ipynb`.

---

## Persiapan

1. Upload `dataset_praktikum2.csv` ke Google Colab (ikon folder di sidebar kiri â†’ Upload).
2. Jalankan sel berikut.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from statsmodels.graphics.mosaicplot import mosaic
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv("dataset_praktikum2.csv")
print("Ukuran data:", df.shape)
print(df.head())
print(df.dtypes)
```

---

# SOAL 1 â€” Data Cleaning

## 1A. Nilai hilang: mean, median, modus

> **Soal:** Cek nilai hilang dengan `isnull().sum()`, lalu tangani kolom `Umur` dan `Gaji` dengan tiga metode: mean, median, modus. Bandingkan hasilnya.

### Jawaban

**Langkah 1 â€” cek nilai hilang**

```python
print("Jumlah nilai hilang per kolom:")
print(df.isnull().sum())
```

**Langkah 2 â€” imputasi dengan tiga metode** (tiap metode memakai salinan `df` sendiri agar tidak saling menimpa)

```python
# Mean
df_mean = df.copy()
df_mean["Umur"] = df_mean["Umur"].fillna(df_mean["Umur"].mean())
df_mean["Gaji"] = df_mean["Gaji"].fillna(df_mean["Gaji"].mean())

# Median
df_median = df.copy()
df_median["Umur"] = df_median["Umur"].fillna(df_median["Umur"].median())
df_median["Gaji"] = df_median["Gaji"].fillna(df_median["Gaji"].median())

# Modus (mode() mengembalikan Series, ambil nilai pertama)
df_mode = df.copy()
df_mode["Umur"] = df_mode["Umur"].fillna(df_mode["Umur"].mode()[0])
df_mode["Gaji"] = df_mode["Gaji"].fillna(df_mode["Gaji"].mode()[0])

# Nilai pengisi yang dipakai tiap metode
print("Nilai pengisi Umur -> mean: %.2f | median: %.2f | modus: %.2f" %
      (df["Umur"].mean(), df["Umur"].median(), df["Umur"].mode()[0]))
print("Nilai pengisi Gaji -> mean: %.2f | median: %.2f | modus: %.2f" %
      (df["Gaji"].mean(), df["Gaji"].median(), df["Gaji"].mode()[0]))
```

**Langkah 3 â€” bandingkan hasilnya**

```python
def ringkas(d, nama):
    return pd.DataFrame({
        "Metode": [nama],
        "Mean_Umur": [d["Umur"].mean()], "Median_Umur": [d["Umur"].median()], "Std_Umur": [d["Umur"].std()],
        "Mean_Gaji": [d["Gaji"].mean()], "Median_Gaji": [d["Gaji"].median()], "Std_Gaji": [d["Gaji"].std()],
    })

perbandingan = pd.concat([
    ringkas(df, "Asli (NaN diabaikan)"),
    ringkas(df_mean, "Imputasi Mean"),
    ringkas(df_median, "Imputasi Median"),
    ringkas(df_mode, "Imputasi Modus"),
], ignore_index=True)
print(perbandingan.round(2).to_string(index=False))
```

**Cara membaca perbandingan:**

| Metode | Karakter | Kapan cocok |
|---|---|---|
| **Mean** | Memakai rata-rata, sehingga **tertarik oleh outlier** (misalnya `Umur = 200` menaikkan rata-rata, seperti terlihat di Modul 2 hlm. 5 saat `Umur` terisi 67,5). Simpangan baku cenderung mengecil karena banyak nilai diisi dengan satu angka yang sama. | Data simetris, tanpa outlier |
| **Median** | Nilai tengah, **tahan terhadap outlier**. | Data miring atau ada outlier |
| **Modus** | Nilai yang paling sering muncul. Pada data kontinu seperti `Gaji`, hampir semua nilai berbeda, sehingga modus bisa berupa nilai sembarang (yang pertama dari nilai-nilai yang frekuensinya sama). | Data kategorik atau diskrit |

**âœï¸ Isi dari hasil run-mu:**

- Jumlah nilai hilang: `Umur` = ___ , `Gaji` = ___ , `Nama` = ___
- Mean Umur = ___ ; Median Umur = ___ ; Modus Umur = ___
- Mean Gaji = ___ ; Median Gaji = ___ ; Modus Gaji = ___
- Kesimpulan: jika mean dan median berbeda jauh, berarti ada outlier atau distribusi miring. Metode yang dipilih = ___ , karena ___ .

**Keputusan untuk langkah selanjutnya:** saya memakai **hasil imputasi median** (`df_median`) sebagai dasar pembersihan berikutnya, karena `Umur` dan `Gaji` biasanya mengandung outlier dan median tidak ikut tertarik olehnya. (Ganti jika hasil perbandinganmu menunjukkan alasan yang berbeda.)

---

## 1B. Outlier: boxplot + IQR

> **Soal:** Deteksi outlier pada `Umur` dan `Gaji` dengan boxplot, lalu tangani dengan metode IQR.

### Jawaban

**Langkah 1 â€” deteksi dengan boxplot** (dibuat terpisah karena skala `Umur` dan `Gaji` sangat berbeda, jika digabung seperti Modul hlm. 7 boxplot `Umur` jadi tidak terbaca)

```python
df_imp = df_median.copy()

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.boxplot(y=df_imp["Umur"], ax=ax[0]); ax[0].set_title("Boxplot Umur (sebelum IQR)")
sns.boxplot(y=df_imp["Gaji"], ax=ax[1]); ax[1].set_title("Boxplot Gaji (sebelum IQR)")
plt.tight_layout()
plt.show()
```

**Langkah 2 â€” tangani dengan IQR**

Aturan: nilai dianggap outlier jika `< Q1 âˆ’ 1,5Ã—IQR` atau `> Q3 + 1,5Ã—IQR`, dengan `IQR = Q3 âˆ’ Q1`.

```python
def batas_iqr(s):
    Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
    IQR = Q3 - Q1
    return Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

mask_aman = pd.Series(True, index=df_imp.index)   # True = bukan outlier
for kol in ["Umur", "Gaji"]:
    bb, ba = batas_iqr(df_imp[kol])
    is_out = (df_imp[kol] < bb) | (df_imp[kol] > ba)
    print(f"{kol}: batas bawah = {bb:.2f}, batas atas = {ba:.2f}, jumlah outlier = {is_out.sum()}")
    mask_aman &= ~is_out

df_iqr = df_imp[mask_aman].copy()
print("Baris sebelum:", len(df_imp), "| sesudah:", len(df_iqr), "| dibuang:", len(df_imp) - len(df_iqr))
```

**Langkah 3 â€” verifikasi dengan boxplot sesudah IQR**

```python
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.boxplot(y=df_iqr["Umur"], ax=ax[0]); ax[0].set_title("Boxplot Umur (sesudah IQR)")
sns.boxplot(y=df_iqr["Gaji"], ax=ax[1]); ax[1].set_title("Boxplot Gaji (sesudah IQR)")
plt.tight_layout()
plt.show()
```

**âœï¸ Isi dari hasil run-mu:**

- Batas IQR Umur: ___ s.d. ___ ; jumlah outlier = ___
- Batas IQR Gaji: ___ s.d. ___ ; jumlah outlier = ___
- Data berkurang dari ___ menjadi ___ baris.

**Catatan:** pada boxplot, titik di luar "kumis" adalah outlier. Metode IQR di sini menghapus barisnya (sesuai Modul hlm. 8). Alternatif lain adalah *capping* (mengganti dengan batas atas/bawah) bila tidak ingin kehilangan data. Setelah penghapusan satu putaran, boxplot bisa menampilkan titik baru di luar kumis, karena Q1/Q3 dan batas berubah. Itu wajar dan tidak perlu diulang.

---

## 1C. Keunikan kolom ID (duplikat)

> **Soal:** Cek apakah kolom `ID` benar-benar unik dengan `duplicated()`. Jelaskan temuannya dan tentukan cara menanganinya.

### Jawaban

**Langkah 1 â€” cek**

```python
print("ID duplikat pada data awal (df)      :", df["ID"].duplicated().sum())
print("ID duplikat pada data saat ini       :", df_iqr["ID"].duplicated().sum())
print("Baris duplikat PENUH pada data ini   :", df_iqr.duplicated().sum())
print("Apakah ID unik?                      :", df_iqr["ID"].is_unique)

# Tampilkan semua baris yang ID-nya muncul lebih dari sekali
dup_id = df_iqr[df_iqr["ID"].duplicated(keep=False)].sort_values("ID")
print(dup_id)

n_id, n_penuh = df_iqr["ID"].duplicated().sum(), df_iqr.duplicated().sum()
if n_id == 0:
    print("\nTemuan: ID sudah unik, tidak ada yang perlu dihapus.")
elif n_id == n_penuh:
    print("\nTemuan: duplikat ID adalah duplikat PENUH (seluruh kolom sama) -> aman dihapus.")
else:
    print("\nTemuan: ada ID sama tetapi isi kolom lain berbeda -> perlu keputusan manual.")
```

**Langkah 2 â€” tangani**

```python
# (a) Hapus baris yang SELURUH kolomnya sama (duplikat penuh), pertahankan yang pertama
df_nodup = df_iqr.drop_duplicates(keep="first")

# (b) Jika ID masih ganda (ID sama, isi berbeda), pertahankan kemunculan pertama
print("ID ganda setelah langkah (a):", df_nodup["ID"].duplicated().sum())
df_nodup = df_nodup.drop_duplicates(subset="ID", keep="first")

print("Baris sebelum:", len(df_iqr), "| sesudah:", len(df_nodup))
print("ID unik sekarang?", df_nodup["ID"].is_unique)
```

**Cara menjelaskan temuan (pilih sesuai hasil run-mu):**

| Temuan | Arti | Penanganan |
|---|---|---|
| ID unik semua | Tidak ada duplikat | Tidak ada tindakan |
| Duplikat **penuh** (semua kolom sama) | Rekaman yang sama tercatat dua kali (kesalahan input/penggabungan) | `drop_duplicates(keep="first")` |
| **ID sama tetapi isi berbeda** | ID bukan kunci unik yang valid; mungkin salah input atau dua orang berbeda terlanjur memakai ID yang sama | Telusuri baris terkait. Bila salah input, simpan satu baris yang benar. Bila dua orang berbeda, beri ID baru. Pada tugas ini dipertahankan kemunculan pertama (`subset="ID", keep="first"`) |

**âœï¸ Isi dari hasil run-mu:** jumlah ID duplikat = ___ ; jenisnya = (penuh / isi berbeda) ; penanganan = ___ ; baris tersisa = ___ .

> Idealnya duplikat dibersihkan **sebelum** imputasi agar nilai ganda tidak ikut memengaruhi mean/median. Urutan di sini mengikuti urutan soal dan alur Modul 2; jika kamu ingin urutan ideal, jalankan langkah 1C pada `df` lebih dulu, lalu ulangi 1A dan 1B.

---

## 1D. Konsistensi penulisan kolom Kota

> **Soal:** Cek konsistensi penulisan pada kolom `Kota` dengan `unique()`, lalu seragamkan dengan mapping.

### Jawaban

**Langkah 1 â€” lihat variasi penulisan**

```python
df_k = df_nodup.copy()
print("Nilai unik Kota (asli):")
print(df_k["Kota"].unique())
print(df_k["Kota"].value_counts())
```

**Langkah 2 â€” rapikan huruf & spasi, lalu lihat lagi variasi yang tersisa**

```python
df_k["Kota"] = df_k["Kota"].astype(str).str.strip().str.lower()
print(sorted(df_k["Kota"].unique()))
```

**Langkah 3 â€” seragamkan dengan mapping.**
Isi `mapping_kota` berdasarkan hasil `unique()` di atas (kunci = bentuk huruf kecil yang ditemukan, nilai = penulisan baku). Contoh di bawah hanya ilustrasi, **sesuaikan dengan variasi yang benar-benar ada di datamu**.

```python
mapping_kota = {
    "jakarta": "Jakarta", "jkt": "Jakarta",
    "bandung": "Bandung", "bdg": "Bandung",
    "surabaya": "Surabaya", "sby": "Surabaya",
    "medan": "Medan", "mdn": "Medan",
    "bandar lampung": "Bandar Lampung", "lampung": "Bandar Lampung", "bdl": "Bandar Lampung",
}

df_k["Kota"] = df_k["Kota"].replace(mapping_kota)
df_k["Kota"] = df_k["Kota"].str.title()     # jaring pengaman untuk nilai yang belum ada di mapping

print("Nilai unik Kota setelah diseragamkan:")
print(df_k["Kota"].value_counts())
```

> **Cek akhir:** semua nilai pada `value_counts()` harus berupa nama kota baku tanpa duplikat makna (misalnya tidak ada lagi "Jkt" dan "Jakarta" bersamaan). Jika masih ada, tambahkan ke `mapping_kota` lalu jalankan ulang sel di atas.

**âœï¸ Isi dari hasil run-mu:** variasi sebelum = ___ ; setelah = ___ (___ kota baku).

### Dataset hasil Soal 1 (dipakai Soal 2â€“5)

```python
df_clean = df_k.reset_index(drop=True)
print("Dataset bersih:", df_clean.shape)
print("Nilai hilang tersisa:\n", df_clean.isnull().sum())
print(df_clean.head())
```

---

# SOAL 2 â€” Data Integration

## 2A. Membuat `df_bonus`

> **Soal:** Buat DataFrame baru `df_bonus` berisi `ID`, `Bonus`, `Departemen` untuk sebagian data.

### Jawaban

`df_bonus` dibuat dari Â±80% ID di `df_clean` (acak), ditambah 2 ID yang **tidak ada** di `df_clean` agar perbedaan jenis join terlihat jelas.

```python
rng = np.random.default_rng(42)

id_ada = rng.choice(df_clean["ID"], size=int(0.8 * len(df_clean)), replace=False)
id_tak_ada = [df_clean["ID"].max() + 100, df_clean["ID"].max() + 101]
id_bonus = np.concatenate([id_ada, id_tak_ada])

df_bonus = pd.DataFrame({
    "ID": id_bonus,
    "Bonus": rng.integers(300, 1001, size=len(id_bonus)),
    "Departemen": rng.choice(["IT", "HR", "Finance", "Marketing"], size=len(id_bonus)),
})
print(df_bonus.shape)
print(df_bonus.head())
```

## 2B. Merge

> **Soal:** Gabungkan dengan dataset hasil Soal 1 memakai `pd.merge()`. Jelaskan jenis join yang dipakai.

### Jawaban

```python
df_merge = pd.merge(df_clean, df_bonus, on="ID", how="inner")

print("df_clean :", df_clean.shape)
print("df_bonus :", df_bonus.shape)
print("df_merge :", df_merge.shape)
print(df_merge.head())

# Pembanding: jika memakai left join
df_left = pd.merge(df_clean, df_bonus, on="ID", how="left")
print("\nJika LEFT join -> baris:", len(df_left), "| baris tanpa Bonus (NaN):", df_left["Bonus"].isnull().sum())
```

**Penjelasan jenis join:** dipakai **inner join** (`how="inner"`) dengan kunci `ID` (sesuai contoh Modul hlm. 9).

- Inner join hanya menyimpan baris yang `ID`-nya **ada di kedua tabel**.
- Baris `df_clean` yang tidak punya data bonus **tidak ikut**, dan baris `df_bonus` yang `ID`-nya tidak ada di `df_clean` (termasuk 2 ID tambahan di atas, atau karyawan yang sudah terbuang saat cleaning) **juga tidak ikut**.
- Hasilnya adalah baris yang lengkap, tanpa NaN pada `Bonus`/`Departemen`.
- Bila ingin mempertahankan seluruh data bersih, gunakan **left join** (`how="left"`), dengan konsekuensi muncul NaN pada `Bonus` dan `Departemen` untuk ID tanpa data bonus.

**âœï¸ Isi dari hasil run-mu:** `df_clean` = ___ baris ; `df_bonus` = ___ baris ; hasil inner = ___ baris.

> **Catatan:** sesuai catatan soal, Soal 3â€“5 tetap memakai **`df_clean`**, bukan `df_merge`/`df_final`, supaya tidak ada baris yang hilang akibat join.

## 2C. Concatenate

> **Soal:** Tambahkan 2â€“3 baris data baru memakai `pd.concat()`.

### Jawaban

```python
kota_valid = df_clean["Kota"].unique()
id_awal = df_clean["ID"].max() + 1

df_baru = pd.DataFrame({
    "ID": [id_awal, id_awal + 1, id_awal + 2],
    "Nama": ["Gilang", "Hana", "Indra"],
    "Umur": [28, 27, 31],
    "Gaji": [4500, 4700, 5200],
    "Kota": [kota_valid[0], kota_valid[1 % len(kota_valid)], kota_valid[2 % len(kota_valid)]],
    "Bonus": [600, 650, 700],
    "Departemen": ["IT", "HR", "Finance"],
})

df_final = pd.concat([df_merge, df_baru], ignore_index=True)
print("Sebelum concat:", df_merge.shape, "| Sesudah:", df_final.shape)
print(df_final.tail(5))
print("ID unik?", df_final["ID"].is_unique)
```

**Penjelasan:** `pd.concat()` menumpuk baris (secara vertikal), jadi kolom `df_baru` harus **sama** dengan `df_merge` (di sini 7 kolom). `ignore_index=True` membuat indeks diurutkan ulang dari 0. ID baru dibuat berlanjut dari ID terbesar agar tetap unik. Data baru dibuat dengan nilai `Kota` yang sudah baku agar tidak merusak hasil Soal 1D.

---

# SOAL 3 â€” Analisis Korelasi

## 3A. Pearson, covariance, heatmap

> **Soal:** Hitung Pearson correlation dan covariance antara `Umur` dan `Gaji`, visualisasikan dengan heatmap.

### Jawaban

```python
num = df_clean[["Umur", "Gaji"]]

korelasi = num.corr(method="pearson")
kovarians = num.cov()
print("Pearson Correlation:\n", korelasi)
print("\nCovariance:\n", kovarians)

plt.figure(figsize=(5, 4))
sns.heatmap(korelasi, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Heatmap Korelasi Pearson: Umur vs Gaji")
plt.show()
```

## 3B. Interpretasi

> **Soal:** Interpretasikan arah dan kekuatan korelasinya.

### Jawaban

Kode berikut menghitung nilai `r`, arah, kekuatan, dan signifikansinya:

```python
r, p_r = stats.pearsonr(df_clean["Umur"], df_clean["Gaji"])
cov_ug = kovarians.loc["Umur", "Gaji"]

arah = "positif" if r > 0 else "negatif"
a = abs(r)
if a < 0.2:   kuat = "sangat lemah"
elif a < 0.4: kuat = "lemah"
elif a < 0.6: kuat = "sedang"
elif a < 0.8: kuat = "kuat"
else:         kuat = "sangat kuat"

print(f"r = {r:.3f} | p-value = {p_r:.4g} | covariance = {cov_ug:.2f}")
print(f"Korelasi {arah} dan {kuat}.")
print("Signifikan secara statistik (alpha 0,05)" if p_r < 0.05 else "Tidak signifikan secara statistik (alpha 0,05)")
```

**Panduan interpretasi:**

| Nilai \|r\| | Kekuatan |
|---|---|
| 0,00 â€“ 0,19 | Sangat lemah |
| 0,20 â€“ 0,39 | Lemah |
| 0,40 â€“ 0,59 | Sedang |
| 0,60 â€“ 0,79 | Kuat |
| 0,80 â€“ 1,00 | Sangat kuat |

- **Arah:** `r > 0` berarti saat `Umur` naik, `Gaji` cenderung ikut naik. `r < 0` berarti sebaliknya.
- **Covariance** memberi arah hubungan (tanda + / âˆ’) tetapi **nilainya bergantung satuan** (di sini umur dalam tahun dan gaji dalam rupiah/dolar), sehingga tidak bisa dibaca kekuatannya. Karena itu kekuatan dibaca dari **korelasi Pearson** yang sudah dinormalisasi ke rentang âˆ’1 s.d. 1.
- Korelasi **bukan** bukti sebab-akibat.

**âœï¸ Isi dari hasil run-mu:** r = ___ ; covariance = ___ ; kesimpulan: korelasi (positif/negatif) yang (kekuatan) antara Umur dan Gaji, artinya ___ .

## 3C. Kelompok_Gaji dan Chi-Square dengan Kota

> **Soal:** Buat kategori `Kelompok_Gaji` (Rendah/Sedang/Tinggi), lalu uji hubungannya dengan `Kota` memakai Chi-Square test.

### Jawaban

**Langkah 1 â€” buat kategori.** Dipakai `pd.qcut` (pembagian berdasarkan **tiga kuantil**), sehingga tiap kelompok berisi jumlah data yang kurang lebih sama.

```python
df_clean["Kelompok_Gaji"] = pd.qcut(df_clean["Gaji"], q=3, labels=["Rendah", "Sedang", "Tinggi"])
print(df_clean["Kelompok_Gaji"].value_counts().sort_index())
print(df_clean.groupby("Kelompok_Gaji", observed=True)["Gaji"].agg(["min", "max"]))
```

**Langkah 2 â€” hipotesis & uji Chi-Square**

- Hâ‚€: `Kota` dan `Kelompok_Gaji` **independen** (tidak berhubungan)
- Hâ‚: `Kota` dan `Kelompok_Gaji` **berhubungan**
- Keputusan: tolak Hâ‚€ jika p-value < 0,05

```python
tabel_kontingensi = pd.crosstab(df_clean["Kota"], df_clean["Kelompok_Gaji"])
print("Tabel Kontingensi:\n", tabel_kontingensi)

chi2, p, dof, expected = stats.chi2_contingency(tabel_kontingensi)
print("\nChi2 Statistic    :", round(chi2, 4))
print("p-value           :", round(p, 4))
print("Degrees of Freedom:", dof)
print("Expected Frequencies:\n", pd.DataFrame(expected, index=tabel_kontingensi.index,
                                                columns=tabel_kontingensi.columns).round(2))
print("Sel dengan expected < 5:", (expected < 5).sum(), "dari", expected.size)

if p < 0.05:
    print("\nKesimpulan: Ada hubungan signifikan antara Kota dan Kelompok_Gaji.")
else:
    print("\nKesimpulan: Tidak ada hubungan signifikan antara Kota dan Kelompok_Gaji.")
```

**Cara membaca hasil:**

- `p < 0,05` â†’ tolak Hâ‚€ â†’ ada hubungan antara kota dan kelompok gaji. `p â‰¥ 0,05` â†’ gagal menolak Hâ‚€ â†’ tidak cukup bukti adanya hubungan.
- **Asumsi uji:** frekuensi harapan (*expected*) idealnya â‰¥ 5 pada sebagian besar sel. Bila banyak sel < 5 (biasanya karena kota terlalu banyak atau sampel kecil), hasil kurang andal. Gabungkan kota yang jarang muncul atau gunakan uji lain.

**âœï¸ Isi dari hasil run-mu:** ChiÂ² = ___ ; dof = ___ ; p-value = ___ ; keputusan = ___ ; sel expected < 5 = ___ .

## 3D. Mosaic plot

> **Soal:** Visualisasikan hasil poin C dengan mosaic plot.

### Jawaban

```python
df_mosaic = df_clean[["Kota", "Kelompok_Gaji"]].astype(str)

plt.figure(figsize=(8, 5))
mosaic(df_mosaic, ["Kota", "Kelompok_Gaji"], title="Mosaic Plot: Kota vs Kelompok_Gaji")
plt.show()
```

**Cara membaca:** lebar kolom = proporsi jumlah data tiap kota; tinggi tiap blok dalam kolom = proporsi kelompok gaji di kota itu. Jika tinggi blok Rendah/Sedang/Tinggi **hampir sama di semua kota**, berarti tidak ada hubungan (konsisten dengan p-value besar). Jika polanya berbeda mencolok antarkota, berarti ada hubungan.

**âœï¸ Isi dari hasil run-mu:** (jelaskan pola yang terlihat, misalnya kota mana yang dominan Tinggi/Rendah, dan apakah sesuai dengan hasil Chi-Square).

---

# SOAL 4 â€” Data Reduction

## 4A. PCA pada Umur dan Gaji

> **Soal:** Lakukan PCA pada `Umur` dan `Gaji`, tampilkan `explained_variance_ratio_` dan scatter plot-nya.

### Jawaban

Skala `Gaji` (ribuan) jauh lebih besar daripada `Umur` (puluhan). PCA peka terhadap skala, jadi dibandingkan **dua versi**: (1) data mentah seperti di modul, dan (2) data yang distandarkan lebih dulu.

```python
X = df_clean[["Umur", "Gaji"]]

# Versi 1: tanpa standarisasi (seperti Modul hlm. 15)
pca_raw = PCA(n_components=2)
X_pca_raw = pca_raw.fit_transform(X)

# Versi 2: dengan standarisasi
X_std = StandardScaler().fit_transform(X)
pca_std = PCA(n_components=2)
X_pca_std = pca_std.fit_transform(X_std)

print("explained_variance_ratio_ (tanpa standarisasi):", pca_raw.explained_variance_ratio_.round(4))
print("explained_variance_ratio_ (dengan standarisasi):", pca_std.explained_variance_ratio_.round(4))
print("\nBobot komponen (tanpa standarisasi):\n", pd.DataFrame(pca_raw.components_, columns=["Umur", "Gaji"], index=["PC1", "PC2"]).round(4))
print("\nBobot komponen (dengan standarisasi):\n", pd.DataFrame(pca_std.components_, columns=["Umur", "Gaji"], index=["PC1", "PC2"]).round(4))

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].scatter(X_pca_raw[:, 0], X_pca_raw[:, 1], color="green", s=20)
ax[0].set_title("Hasil PCA (tanpa standarisasi)")
ax[1].scatter(X_pca_std[:, 0], X_pca_std[:, 1], color="purple", s=20)
ax[1].set_title("Hasil PCA (dengan standarisasi)")
for a_ in ax:
    a_.set_xlabel("Komponen 1"); a_.set_ylabel("Komponen 2"); a_.grid(True)
plt.tight_layout()
plt.show()
```

**Cara membaca:**

- `explained_variance_ratio_` adalah **proporsi varians** yang dijelaskan tiap komponen (jumlahnya 1). Jika Komponen 1 sudah menjelaskan hampir seluruh varians, data 2 dimensi itu bisa diringkas menjadi 1 dimensi tanpa banyak kehilangan informasi.
- **Tanpa standarisasi**, Komponen 1 hampir pasti didominasi `Gaji` (bobotnya mendekati 1) dan rasio variansnya mendekati 100%. Itu karena **skala `Gaji` besar, bukan karena `Gaji` lebih penting**.
- **Dengan standarisasi**, kedua variabel setara. Nilai Komponen 1 kemudian mencerminkan seberapa kuat `Umur` dan `Gaji` berkorelasi. Makin kuat korelasi (Soal 3), makin besar proporsi Komponen 1.
- Pada scatter plot, jika titik membentuk garis memanjang di sepanjang sumbu Komponen 1, itu menandakan Komponen 1 menangkap hampir semua variasi.

**âœï¸ Isi dari hasil run-mu:** tanpa standarisasi: PC1 = ___ , PC2 = ___ ; dengan standarisasi: PC1 = ___ , PC2 = ___ . Kesimpulan: ___ .

## 4B. Histogram Umur dan Gaji

> **Soal:** Buat histogram `Umur` dan `Gaji`, jelaskan bentuk distribusinya.

### Jawaban

```python
for col in ["Umur", "Gaji"]:
    plt.figure(figsize=(6, 4))
    sns.histplot(df_clean[col], kde=True, bins=20, color="skyblue")
    plt.title(f"Distribusi Atribut: {col}")
    plt.xlabel(col); plt.ylabel("Frekuensi"); plt.grid(True)
    plt.show()

    skew = df_clean[col].skew()
    if skew > 0.5:    bentuk = "miring ke kanan (right-skewed): ekor panjang di nilai besar"
    elif skew < -0.5: bentuk = "miring ke kiri (left-skewed): ekor panjang di nilai kecil"
    else:             bentuk = "relatif simetris (mendekati normal)"
    print(f"{col}: mean = {df_clean[col].mean():.2f}, median = {df_clean[col].median():.2f}, skewness = {skew:.2f} -> {bentuk}")
    print(f"   Uji normalitas Shapiro-Wilk p = {stats.shapiro(df_clean[col])[1]:.4f} (p > 0,05: tidak ada bukti menyimpang dari normal)\n")
```

**Cara membaca:**

- Histogram membagi nilai ke dalam rentang (*bin*) dan menghitung frekuensinya. Garis kurva adalah estimasi kepadatan (KDE).
- **Bentuk:** lonceng simetris â†’ mendekati normal; ekor panjang ke kanan â†’ miring positif (mean > median); ekor ke kiri â†’ miring negatif; dua puncak â†’ *bimodal* (kemungkinan ada dua kelompok).
- Kode di atas menghitung *skewness* otomatis: |skew| < 0,5 dianggap kurang lebih simetris.

**âœï¸ Isi dari hasil run-mu:**

- Umur: bentuk = ___ ; mean = ___ ; median = ___ ; skewness = ___
- Gaji: bentuk = ___ ; mean = ___ ; median = ___ ; skewness = ___

## 4C. Sampling 30%

> **Soal:** Lakukan sampling 30% data dengan tiga cara: random biasa, without replacement, with replacement. Bandingkan hasilnya.

### Jawaban

```python
# 1. Simple random sampling (tanpa random_state: hasil berubah tiap dijalankan)
sampel_acak = df_clean.sample(frac=0.3)

# 2. Without replacement (satu baris maksimal terambil satu kali)
sampel_tanpa = df_clean.sample(frac=0.3, random_state=42, replace=False)

# 3. With replacement (baris yang sama boleh terambil berulang kali)
sampel_dengan = df_clean.sample(frac=0.3, random_state=42, replace=True)

def ringkas_sampel(s, nama):
    return {
        "Cara": nama,
        "Jumlah baris": len(s),
        "Indeks ganda": int(s.index.duplicated().sum()),
        "Mean Umur": round(s["Umur"].mean(), 2),
        "Std Umur": round(s["Umur"].std(), 2),
        "Mean Gaji": round(s["Gaji"].mean(), 2),
        "Std Gaji": round(s["Gaji"].std(), 2),
    }

tabel_sampel = pd.DataFrame([
    ringkas_sampel(df_clean, "Populasi (df_clean)"),
    ringkas_sampel(sampel_acak, "Random biasa"),
    ringkas_sampel(sampel_tanpa, "Without replacement"),
    ringkas_sampel(sampel_dengan, "With replacement"),
])
print(tabel_sampel.to_string(index=False))

print("\nIrisan indeks acak vs tanpa pengembalian:", len(set(sampel_acak.index) & set(sampel_tanpa.index)))
print("Irisan indeks tanpa vs dengan pengembalian :", len(set(sampel_tanpa.index) & set(sampel_dengan.index)))
print("\nContoh baris yang terambil lebih dari sekali (with replacement):")
print(sampel_dengan[sampel_dengan.index.duplicated(keep=False)].sort_index().head(10))
```

**Cara membandingkan:**

| Cara | Ciri |
|---|---|
| **Random biasa** (`sample(frac=0.3)`) | Tiap baris punya peluang sama; tanpa `random_state` sampel berubah di setiap run. Secara bawaan pandas memakai `replace=False`. |
| **Without replacement** | Tiap baris terambil **maksimal sekali**, tidak ada indeks ganda. Dengan `random_state=42` hasilnya tetap (reprodusibel). |
| **With replacement** | Baris yang sama bisa **terambil berkali-kali**, sehingga ada indeks ganda (terlihat pada kolom "Indeks ganda" dan contoh di atas). Sebagian baris asli tidak terpilih sama sekali. |

- Ketiganya sama-sama berukuran 30% (jumlah baris sama), tetapi **isinya berbeda** karena mekanisme pengambilan dan seed berbeda (sesuai Modul hlm. 19).
- *Random biasa* dan *without replacement* pada dasarnya mekanisme yang sama. Bedanya hanya *seed* (yang pertama acak setiap run, yang kedua dikunci 42).
- Perbandingan mean dan std sampel terhadap populasi menunjukkan seberapa **representatif** sampelnya. Selisih kecil berarti sampel mewakili populasi dengan baik. *With replacement* cenderung lebih bervariasi karena adanya duplikasi.

**âœï¸ Isi dari hasil run-mu:** jumlah sampel = ___ baris ; indeks ganda with replacement = ___ ; sampel yang paling mendekati mean/std populasi = ___ ; kesimpulan = ___ .

---

# SOAL 5 â€” Data Transformation

Semua normalisasi dilakukan pada kolom `Umur` dan `Gaji` dari `df_clean`.

## 5A. Min-Max Normalization

> **Soal:** Lakukan Min-Max Normalization pada `Umur` dan `Gaji` (manual dan dengan `MinMaxScaler`).

### Jawaban

Rumus: $v' = \dfrac{v - min_A}{max_A - min_A}$ â†’ hasil berada di rentang **[0, 1]**.

```python
X = df_clean[["Umur", "Gaji"]]

# Manual
minmax_manual = (X - X.min()) / (X.max() - X.min())

# MinMaxScaler
minmax_sk = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=X.columns, index=X.index)

print("Manual (5 baris pertama):\n", minmax_manual.head())
print("\nMinMaxScaler (5 baris pertama):\n", minmax_sk.head())
print("\nHasil manual == MinMaxScaler ?", np.allclose(minmax_manual, minmax_sk))
print("Min:", minmax_manual.min().values, "| Max:", minmax_manual.max().values)
```

**Penjelasan:** kedua cara menghasilkan nilai yang sama persis karena rumusnya identik. Nilai terkecil menjadi 0 dan terbesar menjadi 1.

## 5B. Z-Score Normalization

> **Soal:** Lakukan Z-Score Normalization (manual dan dengan `StandardScaler`).

### Jawaban

Rumus: $v' = \dfrac{v - mean(A)}{std(A)}$ â†’ hasil berpusat di 0 dengan simpangan baku 1.

```python
# Manual (pandas .std() memakai ddof=1, simpangan baku SAMPEL, seperti di modul)
z_manual = (X - X.mean()) / X.std()

# Manual dengan ddof=0 (simpangan baku POPULASI, ini yang dipakai StandardScaler)
z_manual_pop = (X - X.mean()) / X.std(ddof=0)

# StandardScaler
z_sk = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)

print("Manual ddof=1 (5 baris pertama):\n", z_manual.head())
print("\nStandardScaler (5 baris pertama):\n", z_sk.head())
print("\nManual ddof=1 == StandardScaler ?", np.allclose(z_manual, z_sk))
print("Manual ddof=0 == StandardScaler ?", np.allclose(z_manual_pop, z_sk))
print("\nMean :", z_sk.mean().round(6).values, "| Std (ddof=0):", z_sk.std(ddof=0).round(6).values)
```

**Penjelasan:** hasil manual dan `StandardScaler` **sedikit berbeda** kalau manual memakai `.std()` bawaan pandas (pembagi nâˆ’1), sedangkan `StandardScaler` memakai pembagi n. Jika manual diubah menjadi `ddof=0`, keduanya identik (terlihat dari dua baris `np.allclose`). Selisihnya mengecil seiring bertambahnya jumlah data. Rata-rata hasil â‰ˆ 0 dan simpangan bakunya = 1. Rumus "manual" pada modul (hlm. 21) memang menghasilkan angka yang sedikit berbeda dari `StandardScaler` karena alasan yang sama.

## 5C. Normalization by Decimal Scaling

> **Soal:** Lakukan Normalization by Decimal Scaling.

### Jawaban

Rumus: $v' = \dfrac{v}{10^j}$, dengan $j$ = bilangan bulat terkecil sehingga $\max(|v'|) < 1$.

`j` dihitung **per kolom** (sesuai definisi: "nilai maksimum absolut dari atribut A"), jadi `Umur` dan `Gaji` mendapat `j` masing-masing.

```python
def decimal_scaling(s):
    max_abs = s.abs().max()
    j = int(np.floor(np.log10(max_abs))) + 1     # j terkecil sehingga max(|v'|) < 1
    return s / (10 ** j), j

dec = pd.DataFrame(index=X.index)
for kol in X.columns:
    dec[kol], j = decimal_scaling(X[kol])
    print(f"{kol}: maks |v| = {X[kol].abs().max():.2f} -> j = {j} -> maks |v'| = {dec[kol].abs().max():.4f}")

print(dec.head())
```

**Penjelasan:** setiap nilai dibagi $10^j$ sehingga titik desimal bergeser dan nilai mutlak maksimum menjadi < 1. Contoh: jika `Umur` maksimum 58 maka `j = 2` (Ã·100); jika `Gaji` maksimum 12.500 maka `j = 5` (Ã·100.000). *Catatan:* contoh di modul (hlm. 22) memakai **satu `j` untuk semua kolom** dari nilai maksimum keseluruhan. Cara itu juga benar, tetapi membuat kolom yang nilainya kecil (seperti `Umur`) menjadi sangat mengecil. Memakai `j` per kolom lebih sesuai definisi.

## 5D. Perbandingan ketiga metode

> **Soal:** Bandingkan ketiga hasil normalisasi dan jelaskan kapan masing-masing metode paling sesuai dipakai.

### Jawaban

```python
def stat(d, nama):
    t = d.agg(["min", "max", "mean", "std"]).T
    t.insert(0, "Metode", nama)
    return t

banding = pd.concat([
    stat(X, "Data asli"),
    stat(minmax_sk, "Min-Max"),
    stat(z_sk, "Z-Score"),
    stat(dec, "Decimal Scaling"),
]).reset_index().rename(columns={"index": "Kolom"})
print(banding.round(4).to_string(index=False))

fig, ax = plt.subplots(1, 4, figsize=(16, 3.5))
for a_, (d, nama) in zip(ax, [(X, "Asli"), (minmax_sk, "Min-Max"), (z_sk, "Z-Score"), (dec, "Decimal Scaling")]):
    sns.boxplot(data=d, ax=a_); a_.set_title(nama)
plt.tight_layout()
plt.show()
```

**Perbandingan:**

| Aspek | Min-Max | Z-Score | Decimal Scaling |
|---|---|---|---|
| Rumus | (v âˆ’ min)/(max âˆ’ min) | (v âˆ’ mean)/std | v / 10^j |
| Rentang hasil | Tepat **[0, 1]** | Tidak terbatas (umumnya âˆ’3 s.d. 3), rata-rata 0, std 1 | **(âˆ’1, 1)**; rentang pasti bergantung pada `j` |
| Pusat data | Tidak digeser ke 0 | Berpusat di 0 | Tidak digeser ke 0 |
| Pengaruh outlier | **Sensitif**: satu nilai ekstrem menekan sebagian besar data ke area sempit | Lebih tahan, tetapi mean/std tetap terpengaruh | Sensitif terhadap nilai maksimum karena `j` ditentukan olehnya |
| Bentuk distribusi | Dipertahankan | Dipertahankan | Dipertahankan |

**Kapan paling sesuai dipakai:**

- **Min-Max** â†’ bila butuh nilai pada rentang tetap [0, 1]. Cocok untuk algoritma berbasis jarak/gradien seperti **KNN, K-Means, jaringan saraf**, dan data tanpa outlier ekstrem (di tugas ini outlier sudah dibuang dengan IQR, sehingga Min-Max aman dipakai).
- **Z-Score** â†’ bila data kira-kira berdistribusi normal, atau ada outlier ringan, dan algoritma mengasumsikan data berpusat di 0 dengan varians sebanding, seperti **PCA, SVM, regresi dengan regularisasi, regresi logistik**. Juga bila ingin membaca nilai sebagai "berapa simpangan baku dari rata-rata".
- **Decimal Scaling** â†’ cara paling sederhana saat yang dibutuhkan hanya mengecilkan skala agar |nilai| < 1 tanpa memerlukan rentang tertentu. Cukup untuk keperluan dasar atau edukasi, tetapi jarang dipakai pada pemodelan karena tidak menyamakan skala antar-kolom seteliti dua metode lainnya.

**âœï¸ Isi dari hasil run-mu:** (rujuk tabel `banding`) Min-Max: Umur ___â€“___ , Gaji ___â€“___ ; Z-Score: mean â‰ˆ 0 , std = 1 ; Decimal Scaling: j(Umur) = ___ , j(Gaji) = ___ . Metode yang paling cocok untuk dataset ini = ___ , karena ___ .

---

## Checklist Pengumpulan

- [ ] Notebook diberi nama `TugasIndividu2_DataMining_NIM.ipynb`
- [ ] Semua sel dijalankan berurutan dan **output tampil**
- [ ] Setiap soal (1Aâ€“5D) punya penjelasan teks berisi angka dari hasil run-mu (bagian âœï¸)
- [ ] Mapping `Kota` sudah disesuaikan dengan variasi yang ada di datamu
- [ ] Soal 3â€“5 memakai `df_clean`
