import sys
import math
import time
import shutil

class TerminalPlasma:
    def __init__(self):
        # Karakter untuk shading (dari gelap ke terang)
        # ASCII Gradient map
        self.chars = " ...',;:clodxkO0KXNWM"
        self.running = True

    def get_term_size(self):
        # Mendapatkan ukuran terminal saat ini
        size = shutil.get_terminal_size()
        return size.columns, size.lines

    def map_val(self, value, in_min, in_max, out_min, out_max):
        # Fungsi mapping angka (seperti map() di Arduino/Processing)
        return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def render(self):
        t = 0
        try:
            # Sembunyikan kursor terminal (ANSI escape code)
            sys.stdout.write("\033[?25l")
            
            while self.running:
                w, h = self.get_term_size()
                # Buffer string untuk satu frame
                buffer = []
                
                # Pindah kursor ke (0,0) tanpa clear screen (lebih efisien)
                buffer.append("\033[H")

                for y in range(h):
                    row = []
                    for x in range(w):
                        # --- THE MATH MAGIC ---
                        # Kombinasi gelombang sinus untuk efek plasma cair
                        # Variabel 't' membuat animasi bergerak
                        
                        v1 = math.sin(x / 10.0 + t)
                        v2 = math.sin((y / 10.0 + t) / 2.0)
                        v3 = math.sin((x / 10.0 + y / 10.0 + t) / 2.0)
                        
                        # Menghitung jarak dari pusat untuk efek radial
                        cx = x - w / 2
                        cy = y - h / 2
                        v4 = math.sin(math.sqrt(cx**2 + cy**2) / 8.0 + t)

                        # Rata-rata intensitas (-1 s/d 1 diremap jadi 0 s/d len(chars))
                        avg = (v1 + v2 + v3 + v4) / 4
                        
                        char_idx = int(self.map_val(avg, -1, 1, 0, len(self.chars) - 1))
                        
                        # Clamp index biar gak error out of range
                        char_idx = max(0, min(char_idx, len(self.chars) - 1))
                        
                        row.append(self.chars[char_idx])
                    
                    # Tambahkan baris ke buffer
                    buffer.append("".join(row))
                
                # Render seluruh frame sekaligus
                sys.stdout.write("".join(buffer))
                sys.stdout.flush()
                
                # Increment waktu/kecepatan animasi
                t += 0.1
                
                # Sedikit delay biar CPU gak jebol 100%
                time.sleep(0.03)

        except KeyboardInterrupt:
            # Handle CTRL+C dengan anggun
            self.cleanup()
        except Exception as e:
            self.cleanup()
            print(f"Error: {e}")

    def cleanup(self):
        # Munculkan kursor lagi dan clear screen saat keluar
        sys.stdout.write("\033[?25h") 
        sys.stdout.write("\033[2J")
        sys.stdout.write("\033[H")
        sys.stdout.flush()
        print("System Halted.")

if __name__ == "__main__":
    sim = TerminalPlasma()
    sim.render()
