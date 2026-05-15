import pygame
import time
import random

# Inisialisasi Pygame
pygame.init()

# Warna (RGB)
putih = (255, 255, 255)
kuning = (255, 255, 102)
hitam = (30, 30, 30) # Abu-abu gelap agar lebih modern
merah = (213, 50, 80)
hijau = (0, 255, 100)
biru_muda = (50, 153, 213)

# Ukuran Layar
lebar = 900
tinggi = 600
layar = pygame.display.set_mode((lebar, tinggi))
pygame.display.set_caption('Snake Game Pro 2026')

jam = pygame.time.Clock()
ukuran_ular = 15 # Diperbesar sedikit agar lebih proporsional
kecepatan_awal = 10

# Font
font_skor = pygame.font.SysFont("consolas", 25)
font_pesan = pygame.font.SysFont("arial", 40, bold=True)

def tampil_skor(skor, level):
    val_skor = font_skor.render(f"Skor: {skor}", True, putih)
    val_level = font_skor.render(f"Level: {level}", True, kuning)
    layar.blit(val_skor, [10, 10])
    layar.blit(val_level, [10, 40])

def gambar_ular(ukuran_ular, daftar_ular):
    for i, x in enumerate(daftar_ular):
        # Bagian kepala warnanya beda dikit
        warna = (0, 200, 0) if i == len(daftar_ular) - 1 else hijau
        pygame.draw.rect(layar, warna, [x[0], x[1], ukuran_ular, ukuran_ular])
        # Kasih border tipis di tiap badan ular
        pygame.draw.rect(layar, hitam, [x[0], x[1], ukuran_ular, ukuran_ular], 1)

def pesan_tengah(msg, warna):
    mesg = font_pesan.render(msg, True, warna)
    rect = mesg.get_rect(center=(lebar/2, tinggi/2))
    layar.blit(mesg, rect)

def gameLoop():
    game_over = False
    game_close = False

    x1 = lebar / 2
    y1 = tinggi / 2

    x1_baru = 0
    y1_baru = 0

    daftar_ular = []
    panjang_ular = 1
    kecepatan_skrg = kecepatan_awal

    makananx = round(random.randrange(0, lebar - ukuran_ular) / float(ukuran_ular)) * ukuran_ular
    makanany = round(random.randrange(0, tinggi - ukuran_ular) / float(ukuran_ular)) * ukuran_ular

    while not game_over:

        while game_close == True:
            layar.fill(hitam)
            pesan_tengah("GAME OVER! C: Main Lagi | Q: Keluar", merah)
            tampil_skor(panjang_ular - 1, (panjang_ular - 1) // 5 + 1)
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        game_over = True
                        game_close = False
                    if event.key == pygame.K_c:
                        gameLoop()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            if event.type == pygame.KEYDOWN:
                # Logika Anti-Suicide: Tidak bisa balik arah seketika
                if event.key == pygame.K_LEFT and x1_baru == 0:
                    x1_baru = -ukuran_ular
                    y1_baru = 0
                elif event.key == pygame.K_RIGHT and x1_baru == 0:
                    x1_baru = ukuran_ular
                    y1_baru = 0
                elif event.key == pygame.K_UP and y1_baru == 0:
                    y1_baru = -ukuran_ular
                    x1_baru = 0
                elif event.key == pygame.K_DOWN and y1_baru == 0:
                    y1_baru = ukuran_ular
                    x1_baru = 0

        if x1 >= lebar or x1 < 0 or y1 >= tinggi or y1 < 0:
            game_close = True
        
        x1 += x1_baru
        y1 += y1_baru
        layar.fill(hitam)
        
        # Gambar Makanan (dibuat bulat agar lebih manis)
        pygame.draw.circle(layar, merah, (int(makananx + ukuran_ular/2), int(makanany + ukuran_ular/2)), ukuran_ular//2)
        
        kepala_ular = [x1, y1]
        daftar_ular.append(kepala_ular)
        
        if len(daftar_ular) > panjang_ular:
            del daftar_ular[0]

        for x in daftar_ular[:-1]:
            if x == kepala_ular:
                game_close = True

        gambar_ular(ukuran_ular, daftar_ular)
        
        skor_skrg = panjang_ular - 1
        level_skrg = (skor_skrg // 5) + 1
        tampil_skor(skor_skrg, level_skrg)

        pygame.display.update()

        # Cek Makan & Update Kecepatan
        if x1 == makananx and y1 == makanany:
            makananx = round(random.randrange(0, lebar - ukuran_ular) / float(ukuran_ular)) * ukuran_ular
            makanany = round(random.randrange(0, tinggi - ukuran_ular) / float(ukuran_ular)) * ukuran_ular
            panjang_ular += 1
            # Tambah kecepatan setiap naik level (tiap 5 skor)
            kecepatan_skrg = kecepatan_awal + (level_skrg * 2)

        jam.tick(kecepatan_skrg)

    pygame.quit()
    quit()

gameLoop()