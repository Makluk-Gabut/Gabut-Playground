; NN 1-16-1 Training (SGD)
; Kompilasi: nasm -f elf64 nn.asm -o nn.o && ld nn.o -o nn

section .data
    ; Training Data (x = 0.5, target = 1.0)
    input_x         dd 0.5
    target_y        dd 1.0
    learning_rate   dd 0.01
    epochs          dq 1000

    ; Bobot & Bias (Inisialisasi sederhana)
    w1              times 16 dd 0.5    ; Hidden Weights (1x16)
    b1              times 16 dd 0.1    ; Hidden Biases
    w2              times 16 dd 0.2    ; Output Weights (16x1)
    b2              dd 0.05            ; Output Bias

section .bss
    h_out           resd 16            ; Output hidden layer (setelah ReLU)
    z_h             resd 16            ; Input hidden layer (sebelum ReLU)
    y_pred          resd 1
    
    ; Gradients
    delta_out       resd 1
    delta_h         resd 16

section .text
    global _start

_start:
    mov rsi, [epochs]

train_loop:
    push rsi                            ; Simpan counter epoch

    ; === 1. FORWARD PASS ===
    ; Hidden Layer: z_h = (x * w1) + b1
    movss xmm0, [input_x]
    xor rcx, rcx
forward_h:
    movss xmm1, [w1 + rcx*4]
    mulss xmm1, xmm0
    addss xmm1, [b1 + rcx*4]
    movss [z_h + rcx*4], xmm1           ; Simpan z untuk backprop deriv
    
    ; ReLU: h_out = max(0, z_h)
    pxor xmm2, xmm2
    maxss xmm1, xmm2
    movss [h_out + rcx*4], xmm1
    
    inc rcx
    cmp rcx, 16
    jne forward_h

    ; Output Layer: y_pred = sum(h_out * w2) + b2
    pxor xmm3, xmm3                     ; Akumulator
    xor rcx, rcx
forward_o:
    movss xmm1, [h_out + rcx*4]
    mulss xmm1, [w2 + rcx*4]
    addss xmm3, xmm1
    inc rcx
    cmp rcx, 16
    jne forward_o
    addss xmm3, [b2]
    movss [y_pred], xmm3

    ; === 2. BACKWARD PASS ===
    ; Error Output: delta_out = (y_pred - target)
    movss xmm0, [y_pred]
    subss xmm0, [target_y]
    movss [delta_out], xmm0             ; xmm0 = dL/dy

    ; Error Hidden: delta_h = (delta_out * w2) * ReLU_deriv(z_h)
    xor rcx, rcx
backward_h:
    movss xmm1, [delta_out]
    mulss xmm1, [w2 + rcx*4]            ; dL/dy * w2
    
    ; ReLU derivative: 1 if z_h > 0 else 0
    movss xmm2, [z_h + rcx*4]
    pxor xmm3, xmm3
    comiss xmm2, xmm3                   ; Bandingkan z_h dengan 0
    ja relu_positive
    pxor xmm1, xmm1                     ; If z_h <= 0, gradien = 0
    jmp store_delta_h
relu_positive:
    ; xmm1 tetap (dikali 1)
store_delta_h:
    movss [delta_h + rcx*4], xmm1
    inc rcx
    cmp rcx, 16
    jne backward_h

    ; === 3. UPDATE WEIGHTS (SGD) ===
    ; w = w - (lr * gradient)
    movss xmm5, [learning_rate]

    ; Update w2: w2 = w2 - (lr * delta_out * h_out)
    xor rcx, rcx
update_w2:
    movss xmm1, [delta_out]
    mulss xmm1, [h_out + rcx*4]         ; Gradient w2
    mulss xmm1, xmm5                    ; lr * Grad
    movss xmm2, [w2 + rcx*4]
    subss xmm2, xmm1                    ; w2 - (lr * grad)
    movss [w2 + rcx*4], xmm2
    inc rcx
    cmp rcx, 16
    jne update_w2

    ; Update w1: w1 = w1 - (lr * delta_h * input_x)
    xor rcx, rcx
update_w1:
    movss xmm1, [delta_h + rcx*4]
    mulss xmm1, [input_x]               ; Gradient w1
    mulss xmm1, xmm5
    movss xmm2, [w1 + rcx*4]
    subss xmm2, xmm1
    movss [w1 + rcx*4], xmm2
    
    ; Update b1 (bias): b1 = b1 - (lr * delta_h)
    movss xmm3, [delta_h + rcx*4]
    mulss xmm3, xmm5
    movss xmm4, [b1 + rcx*4]
    subss xmm4, xmm3
    movss [b1 + rcx*4], xmm4

    inc rcx
    cmp rcx, 16
    jne update_w1

    pop rsi
    dec rsi
    jnz train_loop                      ; Ulangi sampai epoch habis

exit:
    mov eax, 60
    xor edi, edi
    syscall
