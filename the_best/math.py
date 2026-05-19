import random
import math
import time

def evaluasi_hirarki(ekspresi):
    """Evaluasi ekspresi matematika dengan prioritas operasi (* dan / didahulukan)"""
    # Ganti simbol untuk eval Python
    ekspresi = ekspresi.replace('×', '*').replace('÷', '/')
    try:
        # eval() Python sudah menerapkan prioritas operasi yang benar
        hasil = eval(ekspresi)
        if hasil == int(hasil):
            return int(hasil)
        return None  # bukan bilangan bulat
    except:
        return None

def buat_soal_mudah():
    a = random.randint(1, 20)
    b = random.randint(-20, 20)
    jawaban = a + b
    soal = f"{a} + {b}"
    return soal, jawaban

def buat_soal_sedang():
    # 50% 2 variabel, 50% 3 variabel dengan hirarki
    if random.choice([True, False]):
        # 2 variabel
        op = random.choice(['+', '-', '*', '/'])
        if op == '+':
            a = random.randint(1, 50)
            b = random.randint(-40, 50)
            jawaban = a + b
            soal = f"{a} + {b}"
        elif op == '-':
            a = random.randint(1, 50)
            b = random.randint(-40, 50)
            jawaban = a - b
            soal = f"{a} - {b}"
        elif op == '*':
            a = random.randint(1, 30)
            b = random.randint(-20, 30)
            jawaban = a * b
            soal = f"{a} × {b}"
        else:  # bagi
            b = random.randint(-20, 20)
            while b == 0:
                b = random.randint(-20, 20)
            jawaban = random.randint(-15, 15)
            a = b * jawaban
            if a < 1 or a > 50:
                return buat_soal_sedang()
            soal = f"{a} ÷ {b}"
    else:
        # 3 variabel dengan hirarki (prioritas operasi)
        angka = [random.randint(1, 40) for _ in range(3)]
        # buat beberapa angka negatif untuk variabel ke-2/3
        for i in range(1, 3):
            if random.choice([True, False]):
                angka[i] = random.randint(-30, -1)
        
        # Pilih operasi dengan memastikan pembagian bulat
        while True:
            ops = [random.choice(['+', '-', '*', '/']) for _ in range(2)]
            # Buat string ekspresi
            ekspresi = f"{angka[0]} {ops[0]} {angka[1]} {ops[1]} {angka[2]}"
            hasil = evaluasi_hirarki(ekspresi)
            if hasil is not None and -500 <= hasil <= 500:
                return ekspresi.replace('*', '×').replace('/', '÷'), hasil
    return soal, jawaban

def buat_soal_sulit():
    tipe = random.choice(['biasa', 'kurung', 'akar'])
    
    if tipe == 'akar':
        bil_kuadrat = [i*i for i in range(5, 10) if 20 <= i*i <= 90]
        if not bil_kuadrat:
            bil_kuadrat = [25, 36, 49, 64, 81]
        angka = random.choice(bil_kuadrat)
        jawaban = int(math.sqrt(angka))
        soal = f"√{angka}"
        return soal, jawaban
    
    elif tipe == 'kurung':
        while True:
            a = random.randint(20, 70)
            b = random.randint(-50, 50)
            c = random.randint(-50, 50)
            op1 = random.choice(['+', '-', '*'])
            op2 = random.choice(['+', '-', '*', '/'])
            
            ekspresi = f"({a} {op1} {b}) {op2} {c}"
            hasil = evaluasi_hirarki(ekspresi)
            if hasil is not None and -1000 <= hasil <= 1000:
                return ekspresi.replace('*', '×').replace('/', '÷'), hasil
    
    else:  # biasa, 3-5 variabel dengan hirarki
        for _ in range(10):  # coba maksimal 10 kali
            jumlah_var = random.randint(3, 5)
            angka = [random.randint(20, 90) for _ in range(jumlah_var)]
            # Variabel ke-2 dst bisa negatif
            for i in range(1, jumlah_var):
                if random.choice([True, False]):
                    angka[i] = random.randint(-60, -20)
            
            ops = [random.choice(['+', '-', '*', '/']) for _ in range(jumlah_var - 1)]
            
            # Bangun ekspresi
            ekspresi = str(angka[0])
            for i in range(jumlah_var - 1):
                ekspresi += f" {ops[i]} {angka[i+1]}"
            
            hasil = evaluasi_hirarki(ekspresi)
            if hasil is not None and -2000 <= hasil <= 2000:
                return ekspresi.replace('*', '×').replace('/', '÷'), hasil
        
        # fallback
        return buat_soal_sulit()

def main():
    print("="*60)
    print("PROGRAM LATIHAN SOAL MATEMATIKA (PRIORITAS OPERASI YANG BENAR)")
    print("="*60)
    print("\n ATURAN: Perkalian (×) dan pembagian (÷) DIDAHULUKAN")
    print("   Contoh: 5 + 3 × 2 = 5 + 6 = 11 (BUKAN 8 × 2 = 16)\n")
    print("Pilih tingkat kesulitan:")
    print("1. Mudah     (Penjumlahan 2 variabel, angka 1-20, 3 detik)")
    print("2. Sedang    (2-3 variabel, + - × ÷, angka 1-50, 5 detik)")
    print("3. Sulit     (3-5 variabel + kurung + akar, angka 20-90, 10 detik)")
    
    while True:
        try:
            pilih = int(input("\nMasukkan pilihan (1/2/3): "))
            if pilih in [1,2,3]:
                break
            print("Pilihan tidak valid. Masukkan 1, 2, atau 3.")
        except ValueError:
            print("Masukkan angka!")
    
    if pilih == 1:
        waktu = 3
        func_soal = buat_soal_mudah
        tingkat = "Mudah"
    elif pilih == 2:
        waktu = 5
        func_soal = buat_soal_sedang
        tingkat = "Sedang"
    else:
        waktu = 10
        func_soal = buat_soal_sulit
        tingkat = "Sulit"
    
    jumlah_soal = 5
    benar = 0
    
    print(f"\n=== Tingkat {tingkat} ===")
    print(f" Waktu: {waktu} detik/soal | Jumlah: {jumlah_soal} soal\n")
    input("Tekan Enter untuk mulai...")
    
    for i in range(jumlah_soal):
        soal, jawaban = func_soal()
        print(f"\n{'─'*40}")
        print(f"Soal {i+1}: {soal} = ?")
        print(f"{'─'*40}")
        
        start_time = time.time()
        try:
            user_input = input(f"⏱️  {waktu} detik ➜ Jawaban: ")
            elapsed = time.time() - start_time
            
            if elapsed > waktu:
                print(f"WAKTU HABIS! Jawaban: {jawaban}")
                continue
            
            if user_input.strip() == "":
                print(f"Kosong. Jawaban benar: {jawaban}")
                continue
            
            user_ans = int(user_input)
            if user_ans == jawaban:
                print(f"BENAR! ({elapsed:.1f} detik)")
                benar += 1
            else:
                print(f"SALAH! Jawaban: {jawaban}")
        except ValueError:
            print(f"Masukkan angka bulat! Jawaban: {jawaban}")
    
    print("\n" + "="*60)
    print(f"SKOR: {benar}/{jumlah_soal} ({benar/jumlah_soal*100:.0f}%)")
    if benar == jumlah_soal:
        print(" SEMPURNA! ")
    elif benar >= 3:
        print("BAGUS! Tingkatkan terus!")
    else:
        print("COBA LAGI! Pelajari prioritas operasi ya!")
    print("="*60)

if __name__ == "__main__":
    main()
