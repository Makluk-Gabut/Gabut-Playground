# OS Gabut

Proyek **operating system** dari nol yang dibuat pas lagi gabut berat.

Ini bukan Windows, bukan Linux, bukan juga macOS.  
Ini cuma eksperimen low-level gue buat belajar gimana komputer bener-bener jalan dari boot sampe nunjukin tulisan di layar.

### Isinya apa aja?
- `kernel.c` → kernel sederhana (masih tahap awal banget)
- `assembler.asm` → assembly buat bagian low-level
- `Makefile` → buat compile & build
- `linker.id` → linker script

### Status Saat Ini
**Masih berlanjut** (kalo ada waktu luang)

Sekarang baru bisa:
- Boot via bootloader
- Masuk ke kernel
- Print teks ke layar (mungkin)

### Cara Ngebuild & Test (buat yang berani)

``` bash
cd OS
make
```

(Butuh nasm, gcc cross-compiler, qemu, dll. Setupnya agak ribet, ini bukan buat pemula)
Tujuanku
Mau bikin OS sekecil mungkin yang bisa:

Print "Gabut OS" di layar
Handle keyboard
Mungkin suatu saat bisa jalanin program sederhana (semoga)

update: OSnya udah jadi tapi versi 32 bit doang :], kebutuhannya masih sama sih kayak yang di atas, ohh ya nih linknya
        https://github.com/Makluk-Gabut/OS/tree/x86-legacy-version

Kalau lu juga lagi gabut dan suka ngoprek low-level, silakan fork & ikut gabut bareng 😂

Made with ❤️ + kopi + insomnia
by Makluk Gabut
