# Legacy Step 5/6 Development Note (Codex-era)

> **Historical document.** This file is retained as part of the development/AI-use record. It contains Indonesian notes and Codex-era commands and is **not** the current repository runbook. For the current English OpenRouter workflow, use [`README.md`](README.md) and [`PROJECT_REQUIREMENTS_AND_RUN.md`](PROJECT_REQUIREMENTS_AND_RUN.md). Do not use the legacy Codex commands below for the current OpenRouter development run.

Folder kerja hanya `DEL.COSC723/PROJ/01.TRY01/01.CODE`.
Driver PPO-only, `ppo_runtime.py`, dataset, dan ZIP yang sudah diuji tidak diubah.
Tidak ada training, transaksi blockchain, atau submission Slurm dari pekerjaan ini.

## Kenapa perlu arena baru?

Driver PPO-only menjalankan satu pelaku pada satu pasar. Arena pada
`coopmev_arena.py` memberi empat pelaku dompet masing-masing tetapi pool yang
sama. Contoh: A1 membeli lebih dahulu, cadangan pool berubah, lalu proposal C1
harus diperiksa ulang. Jadi, ini adapter tambahan; bukan mengganti model PPO.

`execution_checker.py` menghitung tiga swap secara mandiri. Ia tidak mencari
strategi terbaik. Jika transaksi tidak valid, perubahan saldo/pool tidak
diterapkan. Gas mengurangi profit USD dan tetap memakai account terpisah dari
wallet token, sesuai runtime sumber.

## Aturan prototype yang dipakai

- Setiap putaran: ambil delapan kandidat dengan probabilitas PPO tertinggi.
  Ukuran transaksi tetap berasal dari head PPO untuk route masing-masing.
  Pemeriksaan kelayakan tidak mengubah ukuran atau mencari ukuran pengganti.
- Keempat agent melihat reserve yang sama dan sisa waktu yang sama pada awal
  putaran. Komputer menghitung proposal satu per satu, bukan empat proses
  trading paralel. Ini simulasi putaran serentak, bukan model latensi jaringan.
- Rencana tersimpan mengurutkan kandidat yang masih layak. Jika tidak ada
  kandidat layak dalam delapan pilihan, kandidat PPO pertama dapat diajukan
  lalu ditolak checker. Hal ini tidak otomatis berarti semua peluang habis.
- C1 mengirim route ke C2 hanya pada kondisi yang mengaktifkan pesan.
  C2 menghindari route tersebut jika ada kandidat lain yang layak. Route yang
  berbeda masih bisa memakai pool yang sama; overlap dicatat, bukan otomatis
  dianggap pelanggaran.
- Kesiapan proposal memakai nomor putaran. Jika seri, urutan dasar
  `A1, C1, A2, C2` diputar berdasarkan `(seed + nomor_putaran)`. Urutan tidak
  dipilih berdasarkan profit. B1 memakai bagian urutan yang berlaku untuk
  C1 dan A1 saja.
- Setiap proposal diperiksa pada reserve terbaru sebelum commit. Satu commit
  memperbarui tiga pool dan satu wallet; tidak ada agent lain yang berjalan
  di tengah commit.
- C6 adalah satu budget waktu aktif untuk seluruh arena, bukan 4 x budget.
  Observation, pemeriksaan peluang global, PPO, seleksi kandidat, pesan,
  antrean, checker, commit, dan pencatatan online masuk budget.
  Loading/warmup, pembuatan rencana Codex, pemeriksaan tambahan untuk mencari
  penyebab peluang hilang, serta penyimpanan hasil tidak masuk budget.
  Waktu pemeriksaan tambahan tersebut tetap dilaporkan terpisah.
- Arena berhenti ketika semua agent memenuhi kondisi ekonomi global atau
  C6 habis. Tidak ada batas 200 attempts. Error teknis menggagalkan run dan
  menghasilkan file diagnostik, bukan hasil sukses.

Aturan ini dicatat sebagai `cosc723-shared-arena-development-v1` dan harus
dibekukan sebelum step 7. Jam nyata dipengaruhi beban komputer; pengulangan
episode dengan jam nyata tidak dijanjikan identik sampai digit terakhir.
Tes dengan jam terkontrol memeriksa urutan yang benar-benar dapat diulang.

## Lima kondisi

| Kondisi | Yang berjalan |
|---|---|
| B1 | C1 melawan A1; hanya konteks, bukan perbandingan komunikasi utama. |
| B2 | Empat agent memakai rencana tersimpan, tanpa pesan C1 ke C2. |
| B3 | Sama seperti B2; pesan dikirim dan dibaca formatnya, tetapi isinya tidak dipakai memilih route. |
| Proposed | C2 memakai pesan dan rencana untuk menghindari route C1 ketika ada alternatif layak. |
| PPO-sharing-no-team-plan | Pesan tetap aktif; C1/C2 mengikuti urutan PPO tanpa rencana tim. Rencana A1/A2 tidak berubah. |

`invalid_proposals` berarti tidak layak pada saat pengajuan.
`rejected` berarti checker menolak saat giliran eksekusi.
`stale_state_rejections` berarti semula layak tetapi ditolak setelah keadaan
berubah. `invalid_committed_trades` harus nol. Kegagalan peluang setelah
commit rival dicatat terpisah sebagai `rival_consumed_first`.

## Tes dan cara mencoba

Pakai Python environment yang sudah tersedia. Contoh dari folder ini:

```bash
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/dpc/kunf0127/envs/03.TNSE_02_DRH_260623_02_CHT/bin/python
$PY -B tests/test_execution_checker.py
$PY -B tests/test_codex_plans.py
$PY -B tests/test_coopmev_arena.py
$PY -B coopmev_arena.py --dry-run
$PY -B coopmev_arena.py --fixture-plans --condition all --seed 30
```

Command terakhir memakai **rencana buatan untuk tes, bukan keluaran Codex**.
Default-nya setting asli latency 4 detik dan wallet USD 2.000 per agent.
Hasil tersimpan pada nama unik dalam `runs/`; file lama tidak ditimpa.
Hasil fixture tidak boleh disebut sebagai keberhasilan integrated PPO+Codex.

Untuk menyiapkan input publik tanpa melakukan transaksi:

```bash
$PY -B coopmev_arena.py --export-planning-input plans/public_example.json
```

Input tersebut berisi setting, route, dan pool awal. Tidak ada profit hasil
evaluasi, seed evaluasi, isi checker, atau pesan pribadi tim di dalamnya.
Rencana hanya menyimpan preferensi route dan penjelasan pendek, bukan kode
Python, perintah shell, atau private chain-of-thought.

## Penghubung Codex

`codex_plans.py` memakai CLI 0.154.0 dan login ChatGPT yang sudah tersedia.
Tidak menggunakan API key. Untuk tes pengembangan ini, model yang diminta
adalah `gpt-6-astra`, sesuai konfigurasi kampus saat pekerjaan dilakukan.
Itu ID yang diminta ke CLI, bukan bukti independen mengenai routing backend.

Setiap pemanggilan memakai folder sementara pribadi di `plans/`, tanpa
riwayat percakapan developer. Salinan login sementara berizin 0600 dan dihapus
setelah pemanggilan. Login serta konfigurasi utama tidak ditulis ulang.
Sebelum request model, kode memeriksa konteks prompt dan menjalankan canary:
file publik dapat dibaca, file di luar ruang planner tidak terlihat, dan
penulisan file ditolak. Beberapa profil CLI masih menawarkan `apply_patch`;
karena itu perlindungannya adalah filesystem terisolasi, **bukan klaim bahwa
semua tool berhasil dihilangkan**. Rencana dengan event penggunaan tool ditolak.

Rencana untuk satu setting disimpan per agent dan digabung menjadi bundle.
Bundle memeriksa ID agent, model yang sama, public-input hash, prompt/schema,
dan hash rencana. Preferensi route ganda atau di luar katalog ditolak oleh
validator lokal. Plans yang diimpor manual diberi label berbeda; tidak boleh
disebut hasil Codex yang sudah diverifikasi.

Setelah bundle pengembangan tersedia, command ini **tidak memanggil Codex
lagi**. Ia memuat rencana dan menjalankan PPO pada pasar yang terus berubah:

```bash
$PY -B coopmev_arena.py --plans plans/dev_latency4/bundle.json --condition all --seed 30
```

Untuk membuat satu rencana baru sebelum evaluasi, bentuk command-nya adalah:

```bash
$PY -B codex_plans.py generate --public-input plans/public_latency4.json --agent-id C1 --model gpt-6-astra --live --output plans/new_development/C1.json
```

Ini benar-benar meminta respons model. Jangan menjalankannya untuk mengganti
rencana final setelah melihat profit. File yang sudah ada tidak ditimpa.

**Batas yang masih terbuka:** JSONL CLI menyediakan token dan jumlah turn,
tetapi tidak membuktikan jumlah seluruh request model beserta retry internal.
Karena kontrak menetapkan maksimum 92 request model untuk final planning,
`--final` belum diizinkan. Jumlah CLI invocation tidak dipalsukan menjadi
jumlah request model. Audit pengembangan, termasuk percobaan yang gagal,
tetap disimpan di `plans/attempts/`. Ini perlu diselesaikan sebelum step 7.

## Batas hasil pengembangan

CLI arena sengaja menerima development seeds 30-32 saja. Tidak ada command
eksperimen final seeds 40-49 dalam tahap ini. Lima kondisi dikalikan 23 setting
dan 10 seeds nantinya menghasilkan 1.150 kasus, bukan jumlah yang sudah diuji.

File hasil menyimpan saldo/pool/gas awal-akhir, fingerprint keadaan, profit
masing-masing agent, hitungan keputusan, overlap, penggunaan pesan/rencana,
alasan berhenti, dan hash kode/model/data. Contoh event dibatasi 32 entri agar
tes ringan. Batas ini hanya ukuran catatan contoh, **bukan batas transaksi**;
field `event_sample_is_complete` menjelaskan apakah contoh itu sudah lengkap.

Step 7 masih mencakup pembekuan konfigurasi/rencana, seluruh eksperimen,
analisis paired confidence interval, grafik, serta demo untuk presentasi.
Belum ada klaim komunikasi meningkatkan profit, Codex mengalahkan PPO,
keamanan universal, atau keuntungan pada blockchain nyata.

## Hasil pemeriksaan 18 September 2026

- `runs/step56_verification_final.txt`: 61 tes lulus, termasuk pemeriksaan
  filesystem sandbox nyata, tanpa request model dari unit tests.
- Perbandingan runtime asli versus salinan: 276 keputusan cocok persis pada
  observation, route, amount, reward, wallet, reserve, dan gas. Tes mencakup
  seluruh 23 setting pada seed 30 dan tambahan kasus seed 31/32.
- Empat rencana asli Codex tersedia pada `plans/dev_latency4/`. Ini hanya
  rencana pengembangan untuk setting latency 4 detik, bukan final planning.
- `runs/arena_codex_seed30.json`: kelima kondisi berjalan dengan rencana
  tersebut, matched initial state, dan nol invalid committed trades.
- Profit tim B2 sekitar USD 1.302,10; Proposed sekitar USD 1.302,10;
  kontrol tanpa rencana Codex tim sekitar USD 1.365,62. Ini satu kasus
  pengembangan. Selisih sangat kecil B2 versus Proposed bukan bukti manfaat
  komunikasi, dan hasil ini tidak mendukung klaim Codex selalu membantu.
- `runs/step56_summary.json` merangkum bukti dan hash. Tidak ada eksperimen
  final atau Slurm yang diluncurkan. Semua 18 file dataset tetap identik
  dengan sumber, begitu juga checkpoint PPO yang digunakan.

File baru: `coopmev_arena.py`, `execution_checker.py`, `codex_plans.py`,
`tests/test_coopmev_arena.py`, `tests/test_execution_checker.py`,
`tests/test_codex_plans.py`, serta README ini. Selain itu, satu assertion pada
`tests/test_ppo_runtime.py` lokal diperbaiki: penolakan terakhir yang bertepatan
dengan C6 tidak dihitung sebagai dua keputusan. Runtime, driver PPO-only, ZIP,
dataset, kontrak submitted, dan kode sumber SYMBOL tidak diedit.

---

## 2 October 2026 planner migration note

The active arena import now uses `openrouter_plans.py` for new saved-plan generation/loading. New planning is pinned to OpenRouter `openai/gpt-oss-120b` and reads credentials only from `OPENROUTER_API_KEY`; see `PROJECT_REQUIREMENTS_AND_RUN.md`. Earlier Codex development artifacts and the historical notes above are retained as provenance and must not be relabeled as OpenRouter output. This planner change differs from the wording of the approved contract and should be disclosed/confirmed before the final protocol is frozen.
