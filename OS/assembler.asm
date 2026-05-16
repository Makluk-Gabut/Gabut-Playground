; Konstanta standar untuk Multiboot Header (wajib ada supaya GRUB mau load kernel kita)
MBALIGN  equ  1 << 0            ; Align modules on page boundaries
MEMINFO  equ  1 << 1            ; Minta bootloader ngasih info map memory
FLAGS    equ  MBALIGN | MEMINFO
MAGIC    equ  0x1BADB002        ; Magic number agar GRUB mengenali ini sebagai kernel
CHECKSUM equ -(MAGIC + FLAGS)   ; Checksum matematis

; --- MULTIBOOT HEADER ---
section .multiboot
align 4
    dd MAGIC
    dd FLAGS
    dd CHECKSUM

; --- STACK SETUP ---
; Reservasi memori BSS (Block Started by Symbol) untuk tumpukan (stack).
section .bss
align 16                ; x86 C ABI butuh 16-byte stack alignment
stack_bottom:
resb 16384              ; Alokasikan 16 KiB untuk stack (cukup untuk kernel dasar)
stack_top:

; --- ENTRY POINT ---
section .text
global _start:function (_start.end - _start)
_start:
    ; 1. Inisialisasi Stack Pointer
    ; Arahkan register ESP (Extended Stack Pointer) ke puncak stack kita
    mov esp, stack_top

    ; 2. CRITICAL FIX: Matikan Interrupt Hardware
    ; Hapus flag interrupt. Karena kita belum membuat IDT (Interrupt Descriptor Table),
    ; hardware interrupt (seperti timer atau keyboard real) bakal bikin CPU panik -> Reboot.
    cli 

    ; 3. Lompat ke kode C
    extern kernel_main
    call kernel_main

    ; 4. Fallback Hang Loop (Safety Net)
    ; Kalau fungsi kernel_main() selesai/return (yang seharusnya tidak terjadi di OS),
    ; kita kunci CPU di dalam infinite loop yang tidak memakan daya 100%.
.hang:  
    cli                 ; Pastikan interrupt tetap mati
    hlt                 ; Halt CPU (tunggu sampai ada interrupt, yang mana tidak akan ada)
    jmp .hang           ; Kalau entah gimana hlt tembus, lompat balik ke .hang
.end:
