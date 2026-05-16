# Nama hasil akhir sistem operasi kamu
TARGET = myos.bin

# Kompiler dan alat yang digunakan
CC = i686-elf-gcc
AS = nasm
LD = i686-elf-ld

# Bendera Kompiler C (tidak optimal ;-;)
CFLAGS = -std=gnu99 -ffreestanding -O2 -Wall -Wextra

# Target utama: otomatis jalan pas ngetik 'make'
all: $(TARGET)

# Langkah 1: Kompilasi assembler.asm menjadi file objek objek_asm.o
assembler.o: assembler.asm
	$(AS) -f elf32 assembler.asm -o assembler.o

# Langkah 2: Kompilasi kernel.c menjadi file objek kernel.o
kernel.o: kernel.c
	$(CC) -c kernel.c -o kernel.o $(CFLAGS)

# Langkah 3: Menyatukan (link) semua file objek sesuai aturan linker.ld
$(TARGET): assembler.o kernel.o
	$(CC) -T linker.ld -o $(TARGET) -ffreestanding -O2 -nostdlib assembler.o kernel.o -lgcc

# Target untuk membersihkan file sisa kompilasi
clean:
	rm -f *.o $(TARGET)

# Target untuk menguji langsung di emulator QEMU
run: $(TARGET)
	qemu-system-i386 -kernel $(TARGET)
