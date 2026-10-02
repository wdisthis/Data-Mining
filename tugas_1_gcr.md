# Tugas 1 Data Mining: Cleaning, Korelasi, dan Normalisasi

Dataset: `raw.csv` (data churn pelanggan)

## Peran Kolom

| Kolom | Peran | Keterangan |
|---|---|---|
| CustomerID | ID / pengenal | Tidak dipakai dalam analisis |
| Gender | Nominal | Male / Female |
| ContractType | Nominal | Jenis kontrak |
| InternetService | Nominal | Jenis layanan internet |
| TechSupport | Nominal | Yes / No |
| Churn | Nominal | Yes / No |
| Age | Numerik | Umur pelanggan |
| Tenure | Numerik | Lama berlangganan (bulan) |
| MonthlyCharges | Numerik | Tagihan per bulan |
| TotalCharges | Numerik | Total tagihan |

Syarat tugas: minimal 1 ID, 2 nominal, 2 numerik. Dataset ini memenuhinya (1 ID, 5 nominal, 4 numerik).

**Cara pakai:** jalankan chunk dari atas ke bawah, satu chunk per sel (Jupyter / Google Colab). Letakkan `raw.csv` di folder yang sama dengan notebook. Kalau pakai Colab, upload dulu filenya.

---

## Chunk 1: Import Library

Memuat library yang dipakai sepanjang tugas. Jika belum terpasang, jalankan `pip install pandas numpy scipy matplotlib seaborn openpyxl`.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

pd.set_option("display.max_columns", None)
```

---

## Chunk 2: Load Dataset

Membaca `raw.csv`. Jika hasilnya hanya satu kolom besar, ganti `sep=","` menjadi `sep=";"`.

```python
df = pd.read_csv("raw.csv", sep=",")
print("Jumlah baris & kolom:", df.shape)
df.head()
```

---

## Chunk 3: Definisi Kolom dan Simpan Data Mentah

Mendefinisikan kelompok kolom, lalu menyimpan data **sebelum cleaning** ke `raw.xlsx` (file Excel pertama untuk diupload).

```python
id_col = "CustomerID"
nominal_cols = ["Gender", "ContractType", "InternetService", "TechSupport", "Churn"]
numeric_cols = ["Age", "Tenure", "MonthlyCharges", "TotalCharges"]

df_raw = df.copy()
df_raw.to_excel("raw.xlsx", index=False)
print("raw.xlsx tersimpan")
```

---

## Chunk 4: Eksplorasi Awal

Melihat tipe data, statistik ringkas, dan isi kolom kategori. Perhatikan kolom numerik yang tipenya `object`, itu tanda ada nilai bermasalah di dalamnya.

```python
df.info()
print()
print(df.describe(include="all").T)
```

---

## Chunk 5: Cek Missing Value

Mengecek nilai kosong asli (`NaN`) dan nilai kosong "tersembunyi" seperti spasi, `?`, `N/A`, atau `-`.

```python
print("Missing value (NaN) per kolom:")
print(df.isna().sum())

placeholder = ["", " ", "?", "N/A", "n/a", "NA", "null", "NULL", "-", "none", "None"]
print("\nNilai placeholder per kolom:")
for col in df.columns:
    jumlah = df[col].astype(str).str.strip().isin(placeholder).sum()
    print(f"{col}: {jumlah}")
```

---

## Chunk 6: Cek Duplikat

Mengecek baris yang sama persis dan ID pelanggan yang muncul lebih dari sekali.

```python
print("Baris duplikat penuh :", df.duplicated().sum())
print("CustomerID duplikat  :", df[id_col].duplicated().sum())
df[df[id_col].duplicated(keep=False)].sort_values(id_col).head(10)
```

---

## Chunk 7: Cek Konsistensi Kolom Kategori

Menampilkan semua nilai unik tiap kolom nominal. Cari variasi penulisan seperti huruf besar/kecil berbeda, spasi berlebih, atau typo (misalnya `male`, `MALE`, ` Male`).

```python
for col in nominal_cols:
    print(f"--- {col} ---")
    print(df[col].value_counts(dropna=False))
    print()
```

---

## Chunk 8: Cek Tipe Data Numerik dan Nilai Tidak Valid

Mencari isi kolom numerik yang tidak bisa dikonversi ke angka (misalnya teks atau spasi) serta nilai yang tidak masuk akal.

```python
for col in numeric_cols:
    konversi = pd.to_numeric(df[col], errors="coerce")
    bermasalah = df[konversi.isna() & df[col].notna()][col]
    print(f"{col}: {len(bermasalah)} nilai non-numerik", bermasalah.unique()[:5])

num_check = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
print("\nAge di luar 0-100        :", ((num_check["Age"] <= 0) | (num_check["Age"] > 100)).sum())
print("Tenure negatif           :", (num_check["Tenure"] < 0).sum())
print("MonthlyCharges <= 0      :", (num_check["MonthlyCharges"] <= 0).sum())
print("TotalCharges negatif     :", (num_check["TotalCharges"] < 0).sum())
```

---

## Chunk 9: Cek Outlier (Metode IQR)

Menghitung batas bawah dan atas dengan IQR, lalu menghitung jumlah outlier per kolom. Boxplot membantu melihatnya secara visual.

```python
for col in numeric_cols:
    s = pd.to_numeric(df[col], errors="coerce")
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    batas_bawah, batas_atas = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((s < batas_bawah) | (s > batas_atas)).sum()
    print(f"{col}: batas [{batas_bawah:.2f}, {batas_atas:.2f}] -> {n_out} outlier")

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, col in zip(axes, numeric_cols):
    sns.boxplot(y=pd.to_numeric(df[col], errors="coerce"), ax=ax)
    ax.set_title(col)
plt.tight_layout()
plt.show()
```

---

## Chunk 10: Proses Cleaning

Langkah cleaning, berurutan:

1. Ubah placeholder kosong menjadi `NaN`.
2. Rapikan kolom kategori (hapus spasi berlebih, samakan huruf besar/kecil).
3. Ubah kolom numerik ke tipe angka.
4. Nilai tidak valid (umur tidak wajar, nilai negatif) diubah menjadi `NaN`.
5. Hapus duplikat.
6. Isi missing value: numerik dengan median, kategori dengan modus.
7. Tangani outlier dengan winsorizing (dipotong ke batas IQR).

Jika di Chunk 7 kamu menemukan typo lain, tambahkan ke dictionary `perbaikan`.

```python
df_clean = df.copy()

# 1. Placeholder kosong -> NaN
df_clean = df_clean.replace(placeholder, np.nan)
df_clean = df_clean.apply(lambda c: c.str.strip() if c.dtype == "object" else c)
df_clean = df_clean.replace("", np.nan)

# 2. Rapikan kolom kategori
perbaikan = {
    "Month-To-Month": "Month-to-Month",
    "Month To Month": "Month-to-Month",
    "Dsl": "DSL",
    "Fiber Optic": "Fiber Optic",
    "Fiber-Optic": "Fiber Optic",
}
for col in nominal_cols:
    df_clean[col] = (df_clean[col]
                     .str.replace(r"\s+", " ", regex=True)
                     .str.title()
                     .replace(perbaikan))

# 3. Konversi kolom numerik
for col in numeric_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

# 4. Nilai tidak valid -> NaN
df_clean.loc[(df_clean["Age"] <= 0) | (df_clean["Age"] > 100), "Age"] = np.nan
df_clean.loc[df_clean["Tenure"] < 0, "Tenure"] = np.nan
df_clean.loc[df_clean["MonthlyCharges"] <= 0, "MonthlyCharges"] = np.nan
df_clean.loc[df_clean["TotalCharges"] < 0, "TotalCharges"] = np.nan

# 5. Hapus duplikat dan baris tanpa ID
df_clean = df_clean.dropna(subset=[id_col])
df_clean = df_clean.drop_duplicates()
df_clean = df_clean.drop_duplicates(subset=[id_col], keep="first")

# 6. Isi missing value
# TotalCharges diisi estimasi Tenure x MonthlyCharges bila memungkinkan
estimasi = df_clean["Tenure"] * df_clean["MonthlyCharges"]
df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(estimasi)

for col in numeric_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
for col in nominal_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

# 7. Outlier -> winsorizing dengan batas IQR
for col in numeric_cols:
    q1, q3 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)
    iqr = q3 - q1
    df_clean[col] = df_clean[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

# Age & Tenure kembali ke bilangan bulat
df_clean["Age"] = df_clean["Age"].round().astype(int)
df_clean["Tenure"] = df_clean["Tenure"].round().astype(int)
df_clean[["MonthlyCharges", "TotalCharges"]] = df_clean[["MonthlyCharges", "TotalCharges"]].round(2)

df_clean = df_clean.reset_index(drop=True)
print("Sebelum cleaning:", df_raw.shape)
print("Sesudah cleaning:", df_clean.shape)
```

---

## Chunk 11: Verifikasi Hasil Cleaning

Memastikan tidak ada lagi missing value atau duplikat, kategori sudah konsisten, dan jumlah baris tetap minimal 100.

```python
print("Missing value :", df_clean.isna().sum().sum())
print("Duplikat      :", df_clean.duplicated().sum())
print("Jumlah baris  :", len(df_clean), "(syarat minimal 100)")
print()
for col in nominal_cols:
    print(col, "->", df_clean[col].unique())
print()
df_clean.info()
```

---

## Chunk 12: Simpan Data Bersih

Menyimpan data **sesudah cleaning** ke `clean.xlsx` (file Excel kedua untuk diupload).

```python
df_clean.to_excel("clean.xlsx", index=False)
print("clean.xlsx tersimpan")
df_clean.head()
```

---

## Chunk 13: Korelasi Data Numerik (Pearson)

Mengukur hubungan linear antar kolom numerik. Nilai mendekati +1 atau -1 berarti hubungan kuat, mendekati 0 berarti lemah.

```python
korelasi_num = df_clean[numeric_cols].corr(method="pearson")
print(korelasi_num.round(3))

plt.figure(figsize=(7, 5))
sns.heatmap(korelasi_num, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Korelasi Pearson Data Numerik")
plt.tight_layout()
plt.show()
```

---

## Chunk 14: Korelasi Data Nominal (Chi-Square dan Cramér's V)

Chi-Square menguji apakah dua kolom nominal saling berhubungan (p-value < 0,05 berarti ada hubungan signifikan). Cramér's V mengukur kekuatannya pada skala 0 sampai 1.

```python
def cramers_v(x, y):
    tabel = pd.crosstab(x, y)
    chi2, p, dof, _ = chi2_contingency(tabel)
    n = tabel.values.sum()
    v = np.sqrt(chi2 / (n * (min(tabel.shape) - 1)))
    return chi2, p, v

hasil = []
for i, a in enumerate(nominal_cols):
    for b in nominal_cols[i + 1:]:
        chi2, p, v = cramers_v(df_clean[a], df_clean[b])
        hasil.append({"Variabel 1": a, "Variabel 2": b,
                      "Chi2": round(chi2, 3), "p-value": round(p, 4),
                      "Cramer's V": round(v, 3),
                      "Signifikan (p<0.05)": p < 0.05})

df_chi = pd.DataFrame(hasil).sort_values("Cramer's V", ascending=False)
df_chi
```

---

## Chunk 15: Heatmap Cramér's V

Menampilkan kekuatan hubungan antar kolom nominal dalam bentuk matriks.

```python
matriks_v = pd.DataFrame(1.0, index=nominal_cols, columns=nominal_cols)
for a in nominal_cols:
    for b in nominal_cols:
        if a != b:
            matriks_v.loc[a, b] = cramers_v(df_clean[a], df_clean[b])[2]

plt.figure(figsize=(7, 5))
sns.heatmap(matriks_v.astype(float), annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1)
plt.title("Cramér's V Data Nominal")
plt.tight_layout()
plt.show()
```

---

## Chunk 16: Normalisasi Min-Max

Mengubah skala kolom numerik ke rentang 0 sampai 1 dengan rumus `(x - min) / (max - min)`. Kolom ID dan nominal tidak diubah.

```python
df_minmax = df_clean.copy()
for col in numeric_cols:
    mn, mx = df_clean[col].min(), df_clean[col].max()
    df_minmax[col] = (df_clean[col] - mn) / (mx - mn)

print("Sebelum normalisasi:")
print(df_clean[numeric_cols].describe().loc[["min", "max", "mean"]].round(3))
print("\nSesudah Min-Max:")
print(df_minmax[numeric_cols].describe().loc[["min", "max", "mean"]].round(3))
df_minmax.head()
```

---

## Chunk 17: Normalisasi Z-Score (Opsional)

Alternatif normalisasi dengan rumus `(x - mean) / std`. Hasilnya berpusat di 0 dengan standar deviasi 1. Pakai salah satu saja (Min-Max atau Z-Score) di laporan, sesuai ketentuan tugas.

```python
df_zscore = df_clean.copy()
for col in numeric_cols:
    df_zscore[col] = (df_clean[col] - df_clean[col].mean()) / df_clean[col].std()

print("Sesudah Z-Score:")
print(df_zscore[numeric_cols].describe().loc[["mean", "std"]].round(3))
df_zscore.head()
```

---

## Chunk 18: Visualisasi Sebelum vs Sesudah Normalisasi

Membandingkan distribusi `MonthlyCharges` sebelum dan sesudah normalisasi. Bentuk distribusi sama, hanya skalanya yang berubah. Cocok untuk screenshot laporan.

```python
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df_clean["MonthlyCharges"], kde=True, ax=axes[0])
axes[0].set_title("Sebelum Normalisasi")
sns.histplot(df_minmax["MonthlyCharges"], kde=True, ax=axes[1], color="orange")
axes[1].set_title("Sesudah Min-Max")
plt.tight_layout()
plt.show()
```

---

## Chunk 19: Simpan Hasil Normalisasi (Opsional)

Tugas hanya mewajibkan upload `raw.xlsx`, `clean.xlsx`, dan laporan PDF. File ini hanya cadangan hasil normalisasi.

```python
df_minmax.to_excel("normalized_minmax.xlsx", index=False)
print("normalized_minmax.xlsx tersimpan")
```

---

## Checklist Upload (3 file)

1. `raw.xlsx` (data sebelum cleaning)
2. `clean.xlsx` (data sesudah cleaning)
3. Laporan PDF berisi screenshot proses cleaning (Chunk 5 sampai 12), analisis korelasi (Chunk 13 sampai 15), dan normalisasi (Chunk 16 dan 18), lengkap dengan interpretasi singkat.
