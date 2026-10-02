# Tugas 1 Data Mining: Cleaning, Korelasi, dan Normalisasi

Dataset: `dataset_kotor.csv` (data pelanggan toko online)

## Peran Kolom

| Kolom | Peran | Keterangan |
|---|---|---|
| ID_Pelanggan | ID / pengenal | Tidak dipakai dalam analisis |
| Nama | Pengenal tambahan | Teks bebas, tidak dipakai dalam analisis |
| Jenis_Kelamin | Nominal | Laki-laki / Perempuan |
| Kota | Nominal | Kota domisili pelanggan |
| Kategori_Produk | Nominal | Fashion, Elektronik, Makanan, Kecantikan, Olahraga, Buku |
| Metode_Pembayaran | Nominal | Transfer Bank, E-Wallet, COD, Kartu Kredit |
| Usia | Numerik | Umur pelanggan (tahun) |
| Total_Belanja | Numerik | Total belanja dalam Rupiah |
| Frekuensi_Beli | Numerik | Jumlah transaksi |
| Rating | Numerik | Rating kepuasan (skala 1 sampai 5) |

Syarat tugas: minimal 1 ID, 2 nominal, 2 numerik. Dataset ini memenuhinya (1 ID, 4 nominal, 4 numerik).

**Cara pakai:** jalankan chunk dari atas ke bawah, satu chunk per sel (Jupyter / Google Colab). Letakkan file CSV di folder yang sama dengan notebook. Kalau pakai Colab, upload dulu filenya. Kalau file CSV kamu ganti nama, ubah variabel `NAMA_FILE` di Chunk 2.

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

Membaca CSV dengan `dtype=str` dan `keep_default_na=False`. Artinya semua isi dibaca apa adanya sebagai teks. Tanpa ini, pandas diam-diam mengubah `N/A` menjadi NaN dan menebak tipe kolom, sehingga masalah data tersembunyi dan tidak terlihat saat eksplorasi.

```python
NAMA_FILE = "dataset_kotor.csv"

df = pd.read_csv(NAMA_FILE, sep=",", dtype=str, keep_default_na=False)
print("Jumlah baris & kolom:", df.shape)
df.head(10)
```

---

## Chunk 3: Definisi Kolom, Fungsi Bantu, dan Simpan Data Mentah

Mendefinisikan kelompok kolom dan daftar placeholder kosong. Fungsi `ke_angka` dibuat sekali di sini karena dipakai berulang untuk eksplorasi (Chunk 8, 9) dan cleaning (Chunk 10). Setelah itu data **sebelum cleaning** disimpan ke `raw.xlsx` (file Excel pertama untuk diupload).

Fungsi `ke_angka` menangani format kotor di dataset ini: `Rp 1.182.000`, `348.000` (titik sebagai pemisah ribuan), `31 tahun`, `32.0`, dan `3,6` (koma desimal). Titik hanya dihapus jika diikuti tepat 3 digit, sehingga `32.0` tetap 32.0 sedangkan `348.000` menjadi 348000.

```python
id_col = "ID_Pelanggan"
nominal_cols = ["Jenis_Kelamin", "Kota", "Kategori_Produk", "Metode_Pembayaran"]
numeric_cols = ["Usia", "Total_Belanja", "Frekuensi_Beli", "Rating"]

placeholder = ["", " ", "?", "N/A", "n/a", "NA", "null", "NULL", "-", "none", "None", "nan", "NaN"]

def ke_angka(seri):
    s = seri.astype(str).str.lower().str.strip()
    s = (s.str.replace("rp", "", regex=False)
          .str.replace("tahun", "", regex=False)
          .str.replace(" ", "", regex=False))
    s = s.str.replace(r"\.(?=\d{3}(\D|$))", "", regex=True)  # titik ribuan
    s = s.str.replace(",", ".", regex=False)                 # koma desimal
    return pd.to_numeric(s, errors="coerce")

df_raw = df.copy()
df_raw.to_excel("raw.xlsx", index=False)
print("raw.xlsx tersimpan")
```

---

## Chunk 4: Eksplorasi Awal

Melihat tipe data dan statistik ringkas. Karena dibaca sebagai teks, semua kolom bertipe `object`. Perhatikan kolom numerik (Usia, Total_Belanja, dll): ini tanda bahwa isinya belum bersih dan harus dikonversi.

```python
df.info()
print()
print(df.describe(include="all").T)
```

---

## Chunk 5: Cek Missing Value

Mengecek nilai kosong asli (`NaN`) dan nilai kosong "tersembunyi" seperti string kosong, spasi, `?`, `N/A`, atau `-`. Dengan `keep_default_na=False`, jumlah NaN akan 0 dan semua kekosongan terdeteksi di hitungan placeholder.

```python
print("Missing value (NaN) per kolom:")
print(df.isna().sum())

print("\nNilai placeholder per kolom:")
for col in df.columns:
    jumlah = df[col].astype(str).str.strip().isin(placeholder).sum()
    print(f"{col}: {jumlah}")
```

---

## Chunk 6: Cek Duplikat

Mengecek baris yang sama persis dan ID pelanggan yang muncul lebih dari sekali. Ada duplikat yang hanya berbeda spasi pada nama, sehingga tidak terdeteksi sebagai duplikat penuh sebelum dirapikan.

```python
print("Baris duplikat penuh :", df.duplicated().sum())
print("ID_Pelanggan duplikat:", df[id_col].duplicated().sum())
df[df[id_col].duplicated(keep=False)].sort_values(id_col).head(12)
```

---

## Chunk 7: Cek Konsistensi Kolom Kategori

Menampilkan semua nilai unik tiap kolom nominal. Cari variasi penulisan seperti huruf besar/kecil berbeda, spasi berlebih, singkatan (`L`, `P`, `Jkt`), atau typo (`Fasion`, `Elektronic`, `Ewallet`).

```python
for col in nominal_cols:
    print(f"--- {col} ---")
    print(df[col].value_counts(dropna=False).sort_index())
    print()
```

---

## Chunk 8: Cek Tipe Data Numerik dan Nilai Tidak Valid

Mencari isi kolom numerik yang tidak bisa dikonversi ke angka, lalu mengecek nilai yang tidak masuk akal setelah dikonversi memakai `ke_angka`. Contoh teks bermasalah: `31 tahun`, `Rp 1.182.000`, `3,6`.

```python
for col in numeric_cols:
    asli = df[col].str.strip()
    konversi = ke_angka(df[col])
    tidak_standar = asli[~asli.str.fullmatch(r"-?\d+(\.\d+)?") & ~asli.isin(placeholder)]
    gagal = asli[konversi.isna() & ~asli.isin(placeholder)]
    print(f"{col}: {len(tidak_standar)} format non-standar, contoh: {tidak_standar.unique()[:4]}")
    print(f"      gagal dikonversi: {len(gagal)}")

num_check = df[numeric_cols].apply(ke_angka)
print("\nUsia di luar 17-70        :", ((num_check["Usia"] < 17) | (num_check["Usia"] > 70)).sum())
print("Total_Belanja <= 0        :", (num_check["Total_Belanja"] <= 0).sum())
print("Frekuensi_Beli di luar 1-50:", ((num_check["Frekuensi_Beli"] < 1) | (num_check["Frekuensi_Beli"] > 50)).sum())
print("Rating di luar 1-5        :", ((num_check["Rating"] < 1) | (num_check["Rating"] > 5)).sum())
```

---

## Chunk 9: Cek Outlier (Metode IQR)

Menghitung batas bawah dan atas dengan IQR pada data yang sudah dikonversi ke angka (nilai tidak valid belum dibuang di tahap ini), lalu menghitung jumlah outlier per kolom. Perhatikan `Total_Belanja`: IQR global menandai banyak data sebagai outlier, padahal sebagian besar hanyalah belanja kategori mahal seperti Elektronik. Karena itu IQR juga dihitung per kategori produk, dan hasilnya jauh lebih sedikit (hanya nilai yang benar-benar ekstrem). Boxplot membantu melihatnya secara visual.

```python
for col in numeric_cols:
    s = ke_angka(df[col])
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    batas_bawah, batas_atas = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((s < batas_bawah) | (s > batas_atas)).sum()
    print(f"{col}: batas [{batas_bawah:.2f}, {batas_atas:.2f}] -> {n_out} outlier")

# Total_Belanja sangat dipengaruhi kategori produk (Elektronik wajar jauh lebih mahal dari Buku),
# jadi cek juga outlier per kategori agar data yang wajar tidak ikut terhitung outlier
tb = ke_angka(df["Total_Belanja"])
kat = df["Kategori_Produk"].str.strip().str.lower()
q1g, q3g = tb.groupby(kat).transform(lambda x: x.quantile(0.25)), tb.groupby(kat).transform(lambda x: x.quantile(0.75))
iqrg = q3g - q1g
n_out_grup = ((tb < q1g - 1.5 * iqrg) | (tb > q3g + 1.5 * iqrg)).sum()
print(f"\nTotal_Belanja outlier jika IQR dihitung per kategori: {n_out_grup}")

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, col in zip(axes, numeric_cols):
    sns.boxplot(y=ke_angka(df[col]), ax=ax)
    ax.set_title(col)
plt.tight_layout()
plt.show()
```

---

## Chunk 10: Proses Cleaning

Langkah cleaning, berurutan:

1. Hapus spasi di pinggir semua kolom, lalu ubah placeholder kosong menjadi `NaN`.
2. Rapikan kolom kategori: samakan huruf besar/kecil, lalu petakan singkatan dan typo ke nilai standar.
3. Ubah kolom numerik ke tipe angka dengan `ke_angka`.
4. Nilai tidak valid (usia, rating, frekuensi di luar rentang wajar, belanja negatif) diubah menjadi `NaN`.
5. Hapus duplikat (setelah data dirapikan, supaya duplikat yang beda spasi ikut terdeteksi).
6. Isi missing value: kategori dengan modus, `Total_Belanja` dengan median per `Kategori_Produk`, numerik lain dengan median.
7. Tangani outlier dengan winsorizing (dipotong ke batas IQR). Untuk `Total_Belanja`, batas IQR dihitung per `Kategori_Produk`.

Jumlah perubahan di tiap langkah dicatat di `log` untuk bahan laporan. Jika di Chunk 7 kamu menemukan variasi lain, tambahkan ke dictionary `peta`.

```python
df_clean = df.copy()
log = {}

# 1. Strip spasi dan placeholder -> NaN
for col in df_clean.columns:
    df_clean[col] = df_clean[col].str.strip()
df_clean = df_clean.replace(placeholder, np.nan)
log["Missing value awal (sel)"] = int(df_clean.isna().sum().sum())

# 2. Rapikan kolom kategori
peta = {
    "Jenis_Kelamin": {"l": "Laki-laki", "pria": "Laki-laki", "laki-laki": "Laki-laki",
                      "p": "Perempuan", "wanita": "Perempuan", "perempuan": "Perempuan"},
    "Kota": {"b. lampung": "Bandar Lampung", "jkt": "Jakarta"},
    "Kategori_Produk": {"fasion": "Fashion", "elektronic": "Elektronik"},
    "Metode_Pembayaran": {"ewallet": "E-Wallet", "tf bank": "Transfer Bank", "cod": "COD"},
}
for col in nominal_cols:
    kunci = df_clean[col].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    df_clean[col] = kunci.map(peta[col]).fillna(kunci.str.title())

df_clean["Nama"] = df_clean["Nama"].str.replace(r"\s+", " ", regex=True).str.title()

# 3. Konversi kolom numerik
for col in numeric_cols:
    df_clean[col] = ke_angka(df_clean[col])

# 4. Nilai tidak valid -> NaN
tidak_valid = {
    "Usia": (df_clean["Usia"] < 17) | (df_clean["Usia"] > 70),
    "Total_Belanja": df_clean["Total_Belanja"] <= 0,
    "Frekuensi_Beli": (df_clean["Frekuensi_Beli"] < 1) | (df_clean["Frekuensi_Beli"] > 50),
    "Rating": (df_clean["Rating"] < 1) | (df_clean["Rating"] > 5),
}
for col, mask in tidak_valid.items():
    log[f"Nilai tidak valid {col}"] = int(mask.sum())
    df_clean.loc[mask, col] = np.nan

# 5. Hapus duplikat dan baris tanpa ID
n_awal = len(df_clean)
df_clean = df_clean.dropna(subset=[id_col])
df_clean = df_clean.drop_duplicates()
df_clean = df_clean.drop_duplicates(subset=[id_col], keep="first")
log["Baris duplikat dihapus"] = n_awal - len(df_clean)

# 6. Isi missing value
for col in nominal_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
df_clean["Nama"] = df_clean["Nama"].fillna("Tidak Diketahui")

# Total_Belanja diisi median per kategori produk (lebih masuk akal daripada median global)
df_clean["Total_Belanja"] = df_clean["Total_Belanja"].fillna(
    df_clean.groupby("Kategori_Produk")["Total_Belanja"].transform("median"))
for col in numeric_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# 7. Outlier -> winsorizing dengan batas IQR
# Total_Belanja dihitung per Kategori_Produk, kolom lain secara global
for col in numeric_cols:
    if col == "Total_Belanja":
        g = df_clean.groupby("Kategori_Produk")[col]
        q1, q3 = g.transform(lambda x: x.quantile(0.25)), g.transform(lambda x: x.quantile(0.75))
    else:
        q1, q3 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)
    iqr = q3 - q1
    bawah, atas = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    log[f"Outlier dipotong {col}"] = int(((df_clean[col] < bawah) | (df_clean[col] > atas)).sum())
    df_clean[col] = df_clean[col].clip(bawah, atas)

# Tipe data akhir
for col in ["Usia", "Total_Belanja", "Frekuensi_Beli"]:
    df_clean[col] = df_clean[col].round().astype(int)
df_clean["Rating"] = df_clean["Rating"].round(1)

df_clean = df_clean.reset_index(drop=True)
print("Sebelum cleaning:", df_raw.shape)
print("Sesudah cleaning:", df_clean.shape)
```

---

## Chunk 11: Verifikasi Hasil Cleaning

Memastikan tidak ada lagi missing value atau duplikat, kategori sudah konsisten, nilai numerik berada di rentang wajar, dan jumlah baris tetap minimal 100.

```python
print("Missing value :", df_clean.isna().sum().sum())
print("Duplikat      :", df_clean.duplicated().sum())
print("ID duplikat   :", df_clean[id_col].duplicated().sum())
print("Jumlah baris  :", len(df_clean), "(syarat minimal 100)")
print()
for col in nominal_cols:
    print(col, "->", sorted(df_clean[col].unique()))
print()
print(df_clean[numeric_cols].describe().loc[["min", "max", "mean"]].round(2))
print()
df_clean.info()
```

---

## Chunk 12: Ringkasan Cleaning dan Perbandingan Sebelum vs Sesudah

Menampilkan rekap jumlah perubahan di tiap langkah (bisa disalin ke laporan) dan boxplot `Total_Belanja` sebelum dan sesudah cleaning untuk memperlihatkan efek penanganan outlier.

```python
print("RINGKASAN CLEANING")
for k, v in log.items():
    print(f"- {k}: {v}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.boxplot(y=ke_angka(df_raw["Total_Belanja"]), ax=axes[0])
axes[0].set_title("Total_Belanja Sebelum Cleaning")
sns.boxplot(y=df_clean["Total_Belanja"], ax=axes[1], color="orange")
axes[1].set_title("Total_Belanja Sesudah Cleaning")
plt.tight_layout()
plt.show()
```

---

## Chunk 13: Simpan Data Bersih

Menyimpan data **sesudah cleaning** ke `clean.xlsx` (file Excel kedua untuk diupload).

```python
df_clean.to_excel("clean.xlsx", index=False)
print("clean.xlsx tersimpan")
df_clean.head()
```

---

## Chunk 14: Korelasi Data Numerik (Pearson)

Mengukur hubungan linear antar kolom numerik. Nilai mendekati +1 atau -1 berarti hubungan kuat, mendekati 0 berarti lemah. Pasangan dengan korelasi terkuat ditampilkan terurut untuk memudahkan interpretasi.

```python
korelasi_num = df_clean[numeric_cols].corr(method="pearson")
print(korelasi_num.round(3))

pasangan = (korelasi_num.where(np.triu(np.ones(korelasi_num.shape, dtype=bool), k=1))
            .stack().dropna().rename("r").reset_index())
pasangan.columns = ["Variabel 1", "Variabel 2", "r"]
pasangan = pasangan.reindex(pasangan["r"].abs().sort_values(ascending=False).index)
print("\nUrutan pasangan terkuat:")
print(pasangan.round(3).to_string(index=False))

plt.figure(figsize=(7, 5))
sns.heatmap(korelasi_num, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Korelasi Pearson Data Numerik")
plt.tight_layout()
plt.show()
```

---

## Chunk 15: Korelasi Data Nominal (Chi-Square dan Cramér's V)

Chi-Square menguji apakah dua kolom nominal saling berhubungan (p-value < 0,05 berarti ada hubungan signifikan). Cramér's V mengukur kekuatannya pada skala 0 sampai 1 (sekitar 0,1 lemah, 0,3 sedang, 0,5 ke atas kuat). Kolom `Sel Expected<5 (%)` menunjukkan persentase sel dengan frekuensi harapan di bawah 5. Jika angkanya besar (di atas 20%), hasil Chi-Square kurang bisa dipercaya, biasanya terjadi pada kolom dengan banyak kategori seperti `Kota`.

```python
def cramers_v(x, y):
    tabel = pd.crosstab(x, y)
    chi2, p, dof, expected = chi2_contingency(tabel)
    n = tabel.values.sum()
    v = np.sqrt(chi2 / (n * (min(tabel.shape) - 1)))
    persen_kecil = (expected < 5).mean() * 100
    return chi2, p, v, persen_kecil

def kekuatan(v):
    return "Kuat" if v >= 0.5 else "Sedang" if v >= 0.3 else "Lemah" if v >= 0.1 else "Sangat lemah"

hasil = []
for i, a in enumerate(nominal_cols):
    for b in nominal_cols[i + 1:]:
        chi2, p, v, kecil = cramers_v(df_clean[a], df_clean[b])
        hasil.append({"Variabel 1": a, "Variabel 2": b,
                      "Chi2": round(chi2, 3), "p-value": round(p, 4),
                      "Cramer's V": round(v, 3), "Kekuatan": kekuatan(v),
                      "Signifikan (p<0.05)": p < 0.05,
                      "Sel Expected<5 (%)": round(kecil, 1)})

df_chi = pd.DataFrame(hasil).sort_values("Cramer's V", ascending=False)
df_chi
```

---

## Chunk 16: Heatmap Cramér's V

Menampilkan kekuatan hubungan antar kolom nominal dalam bentuk matriks.

```python
matriks_v = pd.DataFrame(1.0, index=nominal_cols, columns=nominal_cols)
for a in nominal_cols:
    for b in nominal_cols:
        if a != b:
            matriks_v.loc[a, b] = cramers_v(df_clean[a], df_clean[b])[2]

plt.figure(figsize=(8, 5))
sns.heatmap(matriks_v.astype(float), annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1)
plt.title("Cramér's V Data Nominal")
plt.tight_layout()
plt.show()
```

---

## Chunk 17: Normalisasi Min-Max

Mengubah skala kolom numerik ke rentang 0 sampai 1 dengan rumus `(x - min) / (max - min)`. Kolom ID dan nominal tidak diubah. Normalisasi ini penting di dataset ini karena `Total_Belanja` (ratusan ribu) skalanya jauh di atas `Rating` (1 sampai 5).

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

## Chunk 18: Normalisasi Z-Score (Opsional)

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

## Chunk 19: Visualisasi Sebelum vs Sesudah Normalisasi

Membandingkan distribusi `Total_Belanja` sebelum dan sesudah normalisasi. Bentuk distribusi sama, hanya skalanya yang berubah. Cocok untuk screenshot laporan.

```python
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df_clean["Total_Belanja"], kde=True, ax=axes[0])
axes[0].set_title("Sebelum Normalisasi")
sns.histplot(df_minmax["Total_Belanja"], kde=True, ax=axes[1], color="orange")
axes[1].set_title("Sesudah Min-Max")
plt.tight_layout()
plt.show()
```

---

## Chunk 20: Simpan Hasil Normalisasi (Opsional)

Tugas hanya mewajibkan upload `raw.xlsx`, `clean.xlsx`, dan laporan PDF. File ini hanya cadangan hasil normalisasi.

```python
df_minmax.to_excel("normalized_minmax.xlsx", index=False)
print("normalized_minmax.xlsx tersimpan")
```

---

## Checklist Upload (3 file)

1. `raw.xlsx` (data sebelum cleaning)
2. `clean.xlsx` (data sesudah cleaning)
3. Laporan PDF berisi screenshot proses cleaning (Chunk 5 sampai 13), analisis korelasi (Chunk 14 sampai 16), dan normalisasi (Chunk 17 dan 19), lengkap dengan interpretasi singkat.
