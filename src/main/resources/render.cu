

__global__ void render(int width, int height, int size,
            double zoom, double cx, double cy, double cz,
            double* mass, double* density,
            double* x, double* y, double* z,
            double* r, double* g, double* b,
            uint8_t* picture) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < width && iy < height) {
        long pixelPos = (iy * width + ix) * 3;

        double wx = cx + (double) ix / zoom;
        double wy = cy + (double) iy / zoom;

        double height = 1e15;
        for (int i = 0; i < size; i++)
        {
            if (z[i] < height) {
                double dst = sqrt(pow(wx - x[i], 2) + pow(wy - y[i], 2));
                double radius = pow(0.2387324 * mass[i] / density[i], 0.333333333);
                if (dst < radius) {
                    dst /= radius;
                    double coef =  255 * (1 - dst);
                    picture[pixelPos] = (uint8_t) (r[i] * coef);
                    picture[pixelPos + 1] = (uint8_t) (g[i] * coef);
                    picture[pixelPos + 2] = (uint8_t) (b[i] * coef);
                }
            }
        }
    }
}