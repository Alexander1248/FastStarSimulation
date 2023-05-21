
extern "C"
__global__ void accelerate(int size, double dt, double G, double* mass,
            double* x, double* y, double* z,
            double* vx, double* vy, double* vz)  {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        for (int j = 0; j < i; j++) {
            double dx = x[j] - x[i];
            double dy = y[j] - y[i];
            double dz = z[j] - z[i];
            double len = dx * dx + dy * dy + dz * dz;
            double force = G * dt * mass[j] / (len * sqrt(len));
            vx[i] += dx * force;
            vy[i] += dy * force;
            vz[i] += dz * force;
        }
        for (int j = i + 1; j < size; j++) {
             double dx = x[j] - x[i];
            double dy = y[j] - y[i];
            double dz = z[j] - z[i];
            double len = dx * dx + dy * dy + dz * dz;
            double force = G * dt * mass[j] / (len * sqrt(len));
            vx[i] += dx * force;
            vy[i] += dy * force;
            vz[i] += dz * force;
        }
    }
}

extern "C"
__global__ void move(int size, double dt,
            double* x, double* y, double* z,
            double* vx, double* vy, double* vz)  {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        x[i] += vx[i] * dt;
        y[i] += vy[i] * dt;
        z[i] += vz[i] * dt;
    }
}

extern "C"
__global__ void collide(int size, double* mass, double* density,
            double* x, double* y, double* z)  {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int mtxSize = size * (size - 1) / 2;
    if (i < mtxSize) {
        int iy = (sqrt(8.0 * i + 1) - 1) * 0.5;
        int ix = i - (iy + 1) * iy * 0.5;
        iy++;

        double dx = x[ix] - x[iy];
        double dy = y[ix] - y[iy];
        double dz = z[ix] - z[iy];
        double len = dx * dx + dy * dy + dz * dz;

        double rx = pow(0.2387324 * mass[ix] / density[ix], 0.333333333);
        double ry = pow(0.2387324 * mass[iy] / density[iy], 0.333333333);
        if (len < rx + ry) {
            if (mass[ix] > mass[iy]) {
                density[ix] = density[ix] * mass[ix] + density[iy] * mass[iy];

                mass[ix] += mass[iy];
                density[ix] /= mass[ix];

                
                mass[iy] = 0;
                density[iy] = 0;
            }
            else {
                density[iy] = density[ix] * mass[ix] + density[iy] * mass[iy];

                mass[iy] += mass[ix];
                density[iy] /= mass[iy];

                mass[ix] = 0;
                density[ix] = 0;
            }
        }
    }
}