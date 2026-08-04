// Vettori di riferimento K-quant dall'oracolo (fase 4c slice A, gate 2 del design §9).
// Linka la libggml del checkout C1 e chiama DIRETTAMENTE le funzioni upstream
// quantize_q3_K / quantize_q2_K (imatrix NULL = path *_ref, lo stesso che
// llama-quantize usa senza --imatrix) e dequantize_row_q3_K / dequantize_row_q2_K.
// Nessuna patch upstream: solo un main che legge f32 e scrive byte.
//
// Uso: kqref <in.f32> <out.bin>
//   in.f32  = N float32 LE, N multiplo di 256
//   out.bin = [q3k: N/256*110 B][q3k deq: N f32][q2k: N/256*84 B][q2k deq: N f32]
#include <stdio.h>
#include <stdlib.h>
#include "ggml.h"
#include "ggml-quants.h"

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "uso: kqref <in.f32> <out.bin>\n"); return 2; }
    FILE *fi = fopen(argv[1], "rb");
    if (!fi) { perror("in"); return 2; }
    fseek(fi, 0, SEEK_END);
    long bytes = ftell(fi);
    fseek(fi, 0, SEEK_SET);
    long n = bytes / 4;
    if (n % 256) { fprintf(stderr, "N=%ld non multiplo di 256\n", n); return 2; }
    float *x = malloc(n * sizeof(float));
    if (fread(x, sizeof(float), n, fi) != (size_t)n) { fprintf(stderr, "read corta\n"); return 2; }
    fclose(fi);

    long nb = n / 256;
    unsigned char *q3 = malloc(nb * 110);
    unsigned char *q2 = malloc(nb * 84);
    float *d3 = malloc(n * sizeof(float));
    float *d2 = malloc(n * sizeof(float));

    // nrow=1, n_per_row=n: senza quant_weights entrambe instradano sul path _ref
    quantize_q3_K(x, q3, 1, n, NULL);
    quantize_q2_K(x, q2, 1, n, NULL);
    dequantize_row_q3_K((const block_q3_K *)q3, d3, n);
    dequantize_row_q2_K((const block_q2_K *)q2, d2, n);

    FILE *fo = fopen(argv[2], "wb");
    if (!fo) { perror("out"); return 2; }
    fwrite(q3, 1, nb * 110, fo);
    fwrite(d3, sizeof(float), n, fo);
    fwrite(q2, 1, nb * 84, fo);
    fwrite(d2, sizeof(float), n, fo);
    fclose(fo);
    fprintf(stderr, "kqref: N=%ld nb=%ld -> %s\n", n, nb, argv[2]);
    return 0;
}
