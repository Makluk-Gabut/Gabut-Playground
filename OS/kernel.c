#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// --- UTILS ---
// Implementasi string compare untuk shell
int strcmp(const char* s1, const char* s2) {
    while (*s1 && (*s1 == *s2)) { s1++; s2++; }
    return *(const unsigned char*)s1 - *(const unsigned char*)s2;
}

// --- VGA MANAGEMENT ---
static const size_t VGA_WIDTH = 80;
static const size_t VGA_HEIGHT = 25;
uint16_t* vga_buffer = (uint16_t*) 0xB8000;
size_t terminal_row = 0;
size_t terminal_column = 0;
uint8_t terminal_color = 0x0F; // Variabel warna: White on Black

void clear_screen() {
    for (size_t y = 0; y < VGA_HEIGHT; y++) {
        for (size_t x = 0; x < VGA_WIDTH; x++) {
            const size_t index = y * VGA_WIDTH + x;
            vga_buffer[index] = (uint16_t) ' ' | ((uint16_t) terminal_color << 8);
        }
    }
    terminal_row = 0;
    terminal_column = 0;
}

void scroll() {
    // Geser semua baris ke atas 1 tingkat
    for (size_t y = 1; y < VGA_HEIGHT; y++) {
        for (size_t x = 0; x < VGA_WIDTH; x++) {
            vga_buffer[(y - 1) * VGA_WIDTH + x] = vga_buffer[y * VGA_WIDTH + x];
        }
    }
    // Bersihkan baris paling bawah
    for (size_t x = 0; x < VGA_WIDTH; x++) {
        vga_buffer[(VGA_HEIGHT - 1) * VGA_WIDTH + x] = (uint16_t) ' ' | ((uint16_t) terminal_color << 8);
    }
    terminal_row = VGA_HEIGHT - 1;
}

void print_char(char c) {
    if (c == '\n') {
        terminal_column = 0;
        terminal_row++;
    } else if (c == '\b') {
        if (terminal_column > 0) {
            terminal_column--;
            vga_buffer[terminal_row * VGA_WIDTH + terminal_column] = (uint16_t) ' ' | ((uint16_t) terminal_color << 8);
        }
    } else {
        const size_t index = terminal_row * VGA_WIDTH + terminal_column;
        vga_buffer[index] = (uint16_t) c | ((uint16_t) terminal_color << 8);
        terminal_column++;
        
        // Wrap ke baris baru kalau kepanjangan
        if (terminal_column >= VGA_WIDTH) {
            terminal_column = 0;
            terminal_row++;
        }
    }
    
    // Cegah VGA Overflow: Scroll kalau sudah di ujung bawah
    if (terminal_row >= VGA_HEIGHT) {
        scroll();
    }
}

void print_string(const char* str) {
    for (size_t i = 0; str[i] != '\0'; i++) print_char(str[i]);
}

// --- MEMORY MANAGEMENT ---
#define HEAP_SIZE (1024 * 1024)
uint8_t heap[HEAP_SIZE];
uint32_t heap_ptr = 0;

void* malloc(size_t size) {
    // FIX: CPU Alignment (8 byte align)
    heap_ptr = (heap_ptr + 7) & ~7;
    
    // FIX: Bounds checking (cegah overflow)
    if (heap_ptr + size >= HEAP_SIZE) {
        return NULL; 
    }
    
    void* res = &heap[heap_ptr];
    heap_ptr += size;
    return res;
}

// --- IO & KEYBOARD ---
static inline uint8_t inb(uint16_t port) {
    uint8_t ret;
    asm volatile ("inb %1, %0" : "=a"(ret) : "Nd"(port));
    return ret;
}

// Map Scancode sederhana (US QWERTY Lowercase)
const char scancode_to_ascii[] = {
    0, 27, '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '=', '\b',
    '\t', 'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '[', ']', '\n',
    0, 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', '\'', '`', 0,
    '\\', 'z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/', 0, '*', 0, ' '
};

char get_char_from_scancode(uint8_t scancode) {
    if (scancode < sizeof(scancode_to_ascii)) {
        return scancode_to_ascii[scancode];
    }
    return 0;
}

// --- SHELL ---
void execute_command(char* cmd) {
    if (strcmp(cmd, "help") == 0) {
        print_string("Available commands: help, clear, mem\n");
    } else if (strcmp(cmd, "clear") == 0) {
        clear_screen();
    } else if (strcmp(cmd, "mem") == 0) {
        print_string("Memory system active.\n");
    } else if (cmd[0] != '\0') {
        print_string("Command not found: ");
        print_string(cmd);
        print_string("\n");
    }
}

void advanced_shell() {
    char command[80];
    int i = 0;
    
    print_string("MyOS> ");
    
    while (true) {
        if (inb(0x64) & 1) { 
            uint8_t scancode = inb(0x60);
            
            // Abaikan event saat tombol dilepas (key release)
            if (scancode & 0x80) continue;

            char c = get_char_from_scancode(scancode);
            
            if (c == '\n') {
                print_char('\n');
                command[i] = '\0'; // FIX: Null terminator string
                execute_command(command);
                i = 0;
                print_string("MyOS> ");
            } else if (c == '\b') {
                // FIX: Backspace handling
                if (i > 0) {
                    i--;
                    print_char('\b');
                }
            } else if (c != 0) {
                // FIX: Buffer overflow protection di input
                if (i < 79) { 
                    command[i++] = c;
                    print_char(c);
                }
            }
        } else {
            // FIX: Hemat CPU daripada 100% loop buta
            // Kalau interrupt belum nyala, ini bahaya bisa hang total
            // Tapi pause/hlt baik kalau interrupt timer (PIT) sudah setup
            asm volatile ("pause"); 
        }
    }
}

void kernel_main(void) {
    // Clear screen pas boot supaya bersih
    clear_screen();
    print_string("Kernel v2.0 - Security Patched & Upgraded\n");
    print_string("=========================================\n");
    advanced_shell();
}
